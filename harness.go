package sleipnir

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"

	anyllm "github.com/mozilla-ai/any-llm-go"
)

// subAgentTool is a marker interface for tools that represent sub-agents.
// dispatchInParallel detects this interface to invoke runLoop recursively
// instead of calling Invoke directly.
type subAgentTool interface {
	Tool
	subAgentName() string
}

// agentAsToolImpl is the concrete type returned by AgentAsTool.
// Its Invoke panics because the harness routes sub-agent calls through runLoop,
// not through Invoke.
type agentAsToolImpl struct {
	def  ToolDefinition
	name string
}

func (a *agentAsToolImpl) Definition() ToolDefinition { return a.def }
func (a *agentAsToolImpl) subAgentName() string       { return a.name }
func (a *agentAsToolImpl) Invoke(_ context.Context, _ json.RawMessage) (ToolResult, error) {
	panic("sleipnir: agentAsToolImpl.Invoke called directly; use Harness.dispatchInParallel")
}

var _ Tool        = (*agentAsToolImpl)(nil)
var _ subAgentTool = (*agentAsToolImpl)(nil)

type RunInput struct {
	AgentName                 string
	Prompt                    string
	Input                     any
	History                   []anyllm.Message
	Router                    ModelRouter
	Events                    Sink
	HITL                      HITLHandler
	ExtraTools                []Tool
	OmitExtraToolsInheritance bool // false (default) = sub-agents inherit ExtraTools
	MaxTotalTokens            int
}

type RunOutput struct {
	Text     string
	Messages []anyllm.Message
	Usage    Usage
	Stopped  StopReason
}

type HITLHandler interface {
	AskUser(ctx context.Context, agent, question, contextBlurb string) (string, error)
}

type Usage struct {
	InputTokens  int64
	OutputTokens int64
	TotalTokens  int64
}

type StopReason string

const (
	StopDone            StopReason = "done"
	StopIterationBudget StopReason = "iteration_budget"
	StopTokenBudget     StopReason = "token_budget"
	StopHITLTimeout     StopReason = "hitl_timeout"
	StopHITLCancelled   StopReason = "hitl_cancelled"
)

// LLMRequest is the request passed to the LLM. Middleware may mutate fields.
// The Agent field must not be mutated by middleware.
type LLMRequest struct {
	Agent    AgentInfo
	Messages []anyllm.Message
	Tools    []anyllm.Tool // converted from []ToolDefinition via toolsToAnyllm()
	Model    ModelConfig
}

// LLMResponse wraps the provider response and carries AgentInfo for observers.
type LLMResponse struct {
	Agent   AgentInfo
	Message anyllm.Message // the assistant turn (Role == RoleAssistant)
	Usage   Usage          // converted from anyllm.Usage, TotalTokens computed
}

type Harness struct {
	cfg    Config
	agents sync.Map    // map[string]AgentSpec - used for both modes
	frozen atomic.Bool // freeze latch; set on first run when !cgf.AllowLateRegistration
}

func NewHarness(cfg Config) (*Harness, error) {
	cfg = resolveDefaults(cfg)
	if err := validateConfig(cfg); err != nil {
		return nil, err
	}
	return &Harness{cfg: cfg}, nil
}

func validateConfig(cfg Config) error {
	if cfg.CompactThreshold <= 0.0 || cfg.CompactThreshold > 1.0 {
		return fmt.Errorf("sleipnir: CompactThreshold must be in (0.0, 1.0], got %v", cfg.CompactThreshold)
	}
	if cfg.LogFormat != "json" && cfg.LogFormat != "text" {
		return fmt.Errorf("sleipnir: LogFormat must be \"json\" or \"text\", got %q", cfg.LogFormat)
	}
	return nil
}

// Register agent with harness. Returns ErrHarnessFrozen if `h.frozen.Load() && !h.cfg.AllowLateRegistration`
func (h *Harness) RegisterAgent(spec AgentSpec) error {
	// Check if harness is frozen
	if h.frozen.Load() && !h.cfg.AllowLateRegistration {
		return ErrHarnessFrozen
	}

	// Ensure spec.Name is not empty
	if spec.Name == "" {
		return errors.New("sleipnir: agent name empty")
	}

	// Validate tool names
	if e := validateToolNames(spec.Tools); e != nil {
		return ErrToolNameCollision
	}

	// Resolve zeros in agent config -- the stored spec must contain no zeros
	if spec.MaxIterations == 0 {
		spec.MaxIterations = h.cfg.DefaultMaxIterations
	}
	if spec.MaxParallelTools == 0 {
		spec.MaxParallelTools = h.cfg.DefaultMaxParallelTools
	}

	h.agents.Store(spec.Name, spec)

	return nil
}

// AgentAsTool returns a Tool that, when dispatched by the harness, recursively
// runs the named sub-agent. The tool's Definition is derived from the
// registered AgentSpec (Name, Description, InputSchema).
//
// Returns ErrAgentNotRegistered if no agent with that name has been registered.
func (h *Harness) AgentAsTool(name string) (Tool, error) {
	v, ok := h.agents.Load(name)
	if !ok {
		return nil, fmt.Errorf("%w: %q", ErrAgentNotRegistered, name)
	}
	spec := v.(AgentSpec)
	return &agentAsToolImpl{
		def: ToolDefinition{
			Name:        spec.Name,
			Description: spec.Description,
			InputSchema: spec.InputSchema,
		},
		name: spec.Name,
	}, nil
}

// Run the agent loop
func (h *Harness) Run(ctx context.Context, in RunInput) (RunOutput, error) {
	if !h.cfg.AllowLateRegistration {
		h.frozen.Store(true)
	}
	raw, ok := h.agents.Load(in.AgentName)
	if !ok {
		return RunOutput{}, ErrAgentNotRegistered
	}
	spec := raw.(AgentSpec)

	return h.runLoop(ctx, spec, in, "", 0)
}

func (h *Harness) runLoop(ctx context.Context, spec AgentSpec, in RunInput, parentName string, depth int) (RunOutput, error) {
	agentInfo := AgentInfo{
		Name:       spec.Name,
		ParentName: parentName,
		Depth:      depth,
		IsSubAgent: depth > 0,
	}

	allTools := make([]Tool, 0, len(spec.Tools)+len(in.ExtraTools))
	allTools = append(allTools, spec.Tools...)
	allTools = append(allTools, in.ExtraTools...)

	if err := validateToolNames(allTools); err != nil {
		return RunOutput{}, err
	}

	toolMap := make(map[string]Tool, len(allTools))
	for _, t := range allTools {
		toolMap[t.Definition().Name] = t
	}

	emit(in.Events, AgentStartEvent{AgentName: spec.Name, ParentName: parentName})

	history := make([]anyllm.Message, 0, len(in.History)+8)

	if sp := spec.SystemPrompt; sp != nil {
		ai := AgentInput{Prompt: in.Prompt, History: in.History}
		history = append(history, anyllm.Message{
			Role:    anyllm.RoleSystem,
			Content: sp(ai),
		})
	}

	// Unpack in.History and append to the active history
	history = append(history, in.History...)

	if in.Prompt != "" {
		history = append(history, anyllm.Message{
			Role:    anyllm.RoleUser,
			Content: in.Prompt,
		})
	}

	var totalUsage Usage

	for i := 0; i < spec.MaxIterations; i++ {
		modelCfg, err := in.Router.Resolve(ctx, spec.Name)
		if err != nil {
			return RunOutput{}, err
		}

		req := LLMRequest{
			Agent:    agentInfo,
			Messages: history,
			Tools:    toolsToAnyllm(allTools),
			Model:    modelCfg,
		}

		resp, err := callProvider(ctx, req)
		if err != nil {
			return RunOutput{}, err
		}

		totalUsage.InputTokens += resp.Usage.InputTokens
		totalUsage.OutputTokens += resp.Usage.OutputTokens
		totalUsage.TotalTokens += resp.Usage.TotalTokens

		if resp.Message.Content != "" {
			emit(in.Events, TokenEvent{AgentName: spec.Name, Text: resp.Message.ContentString()})
		}
		if resp.Message.Reasoning != nil && resp.Message.Reasoning.Content != "" {
			emit(in.Events, ThinkingEvent{AgentName: spec.Name, Text: resp.Message.Reasoning.Content})
		}

		history = append(history, resp.Message)

		if len(resp.Message.ToolCalls) == 0 {
			out := RunOutput{
				Text:     resp.Message.ContentString(),
				Messages: history,
				Usage:    totalUsage,
				Stopped:  StopDone,
			}
			emit(in.Events, AgentEndEvent{AgentName: spec.Name, Usage: out.Usage, Stopped: StopDone})
			return out, nil
		}

		toolResults := dispatchInParallel(ctx, resp.Message.ToolCalls, toolMap, spec.Name, depth, in, h, in.Events, spec.MaxParallelTools)
		history = append(history, toolResultMessages(toolResults)...)
	}

	emit(in.Events, AgentEndEvent{AgentName: spec.Name, Usage: totalUsage, Stopped: StopIterationBudget})
	return RunOutput{Messages: history, Usage: totalUsage, Stopped: StopIterationBudget}, ErrIterationBudget
}

type toolCallResult struct {
	callID   string
	result   ToolResult
	infraErr error
}

func dispatchInParallel(
	ctx context.Context,
	calls []anyllm.ToolCall,
	toolMap map[string]Tool,
	agentName string,
	depth int,
	parentIn RunInput,
	h *Harness,
	sink Sink,
	maxParallel int,
) []toolCallResult {
	results := make([]toolCallResult, len(calls))
	sem := make(chan struct{}, maxParallel)
	var wg sync.WaitGroup

	for i, call := range calls {
		wg.Add(1)
		go func(i int, call anyllm.ToolCall) {
			defer wg.Done()

			select {
			case sem <- struct{}{}:
			case <-ctx.Done():
				results[i] = toolCallResult{
					callID:   call.ID,
					result:   ToolResult{IsError: true, Content: "context cancelled"},
					infraErr: ctx.Err(),
				}
				return
			}
			defer func() { <-sem }()

			emit(sink, ToolCallEvent{
				AgentName:  agentName,
				ToolCallID: call.ID,
				ToolName:   call.Function.Name,
				Args:       json.RawMessage(call.Function.Arguments),
			})

			tool, ok := toolMap[call.Function.Name]
			var res ToolResult
			var infraErr error
			if !ok {
				res = ToolResult{IsError: true, Content: "unknown tool: " + call.Function.Name}
			} else if sat, isSAT := tool.(subAgentTool); isSAT {
				childIn := buildChildRunInput(parentIn, sat.subAgentName(), json.RawMessage(call.Function.Arguments))
				childSpecRaw, specFound := h.agents.Load(sat.subAgentName())
				if !specFound {
					res = ToolResult{IsError: true, Content: "sub-agent not registered: " + sat.subAgentName()}
				} else {
					childOut, childErr := h.runLoop(ctx, childSpecRaw.(AgentSpec), childIn, agentName, depth+1)
					res, infraErr = subAgentResultToToolResult(childOut, childErr)
				}
			} else {
				res, infraErr = tool.Invoke(ctx, json.RawMessage(call.Function.Arguments))
				if infraErr != nil {
					res = ToolResult{IsError: true, Content: "tool execution failed: " + infraErr.Error()}
					emit(sink, ErrorEvent{AgentName: agentName, Err: infraErr})
				}
			}

			emit(sink, ToolResultEvent{
				AgentName:  agentName,
				ToolCallID: call.ID,
				Result:     res.Content,
				IsError:    res.IsError,
			})
			results[i] = toolCallResult{callID: call.ID, result: res, infraErr: infraErr}
		}(i, call)
	}
	wg.Wait()
	return results
}

// buildChildRunInput constructs a RunInput for a sub-agent call, inheriting
// the parent's router, events, HITL, and (optionally) ExtraTools.
func buildChildRunInput(parent RunInput, agentName string, args json.RawMessage) RunInput {
	child := RunInput{
		AgentName:                 agentName,
		Input:                     args,
		Router:                    parent.Router,
		Events:                    parent.Events,
		HITL:                      parent.HITL,
		OmitExtraToolsInheritance: parent.OmitExtraToolsInheritance,
	}
	if !parent.OmitExtraToolsInheritance {
		child.ExtraTools = parent.ExtraTools
	}
	return child
}

// subAgentResultToToolResult converts a runLoop result from a sub-agent into a
// ToolResult for the parent's history. Budget errors are surfaced as
// IsError=true (the parent can react). Context errors propagate as infraErr so
// the parent run terminates.
func subAgentResultToToolResult(out RunOutput, err error) (ToolResult, error) {
	if err == nil {
		return ToolResult{Content: out.Text}, nil
	}
	if errors.Is(err, ErrIterationBudget) || errors.Is(err, ErrTokenBudget) {
		// Include the stop reason string so callers can identify the budget type.
		return ToolResult{IsError: true, Content: string(out.Stopped) + ": " + err.Error()}, nil
	}
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return ToolResult{}, err
	}
	return ToolResult{IsError: true, Content: "sub-agent failed: " + err.Error()}, nil
}

func toolResultMessages(results []toolCallResult) []anyllm.Message {
	msgs := make([]anyllm.Message, len(results))
	for i, r := range results {
		msgs[i] = anyllm.Message{
			Role:       anyllm.RoleTool,
			Content:    r.result.Content,
			ToolCallID: r.callID,
		}
	}
	return msgs
}

func emit(sink Sink, e Event) {
	if sink != nil {
		sink.Send(e)
	}
}

func callProvider(ctx context.Context, req LLMRequest) (LLMResponse, error) {
	if err := ctx.Err(); err != nil {
		return LLMResponse{}, err
	}
	params := anyllm.CompletionParams{
		Model:           req.Model.Model,
		Messages:        req.Messages,
		Tools:           req.Tools,
		ReasoningEffort: req.Model.ReasoningEffort,
	}
	if req.Model.Temperature != nil {
		params.Temperature = req.Model.Temperature
	}
	if req.Model.MaxOutputTokens != nil {
		params.MaxTokens = req.Model.MaxOutputTokens
	}

	cc, err := req.Model.Provider.Completion(ctx, params)
	if err != nil {
		return LLMResponse{}, err
	}
	if len(cc.Choices) == 0 {
		return LLMResponse{}, fmt.Errorf("sleipnir: provider returned empty choices")
	}

	msg := cc.Choices[0].Message
	var usage Usage
	if cc.Usage != nil {
		usage = Usage{
			InputTokens:  int64(cc.Usage.PromptTokens),
			OutputTokens: int64(cc.Usage.CompletionTokens),
			TotalTokens:  int64(cc.Usage.TotalTokens),
		}
	}
	return LLMResponse{Agent: req.Agent, Message: msg, Usage: usage}, nil
}

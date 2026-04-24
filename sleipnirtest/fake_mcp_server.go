package sleipnirtest

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

// FakeMCPServer is an in-process MCP server for testing.
// Register tools with AddTool, then call Start to connect a client.
// After Start returns, Client() provides a connected *mcp.ClientSession.
type FakeMCPServer struct {
	t      *testing.T
	server *mcp.Server
	client *mcp.ClientSession
}

// NewFakeMCPServer creates a FakeMCPServer. Call AddTool to register tools,
// then call Start to connect. Cleanup is registered with t.Cleanup.
func NewFakeMCPServer(t *testing.T) *FakeMCPServer {
	t.Helper()
	s := mcp.NewServer(&mcp.Implementation{Name: "fake-mcp-server", Version: "v0.0.1"}, nil)
	return &FakeMCPServer{t: t, server: s}
}

// AddTool registers a low-level tool on the server. fn receives the raw
// arguments as a map[string]any and returns a text result.
// Must be called before Start.
func (f *FakeMCPServer) AddTool(name, description string, fn func(ctx context.Context, args map[string]any) (string, error)) {
	f.t.Helper()
	inputSchema := json.RawMessage(`{"type":"object"}`)
	f.server.AddTool(
		&mcp.Tool{
			Name:        name,
			Description: description,
			InputSchema: inputSchema,
		},
		func(ctx context.Context, req *mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			var args map[string]any
			if len(req.Params.Arguments) > 0 {
				if err := json.Unmarshal(req.Params.Arguments, &args); err != nil {
					return nil, err
				}
			}
			text, err := fn(ctx, args)
			if err != nil {
				return &mcp.CallToolResult{
					Content: []mcp.Content{&mcp.TextContent{Text: err.Error()}},
					IsError: true,
				}, nil
			}
			return &mcp.CallToolResult{
				Content: []mcp.Content{&mcp.TextContent{Text: text}},
			}, nil
		},
	)
}

// AddRawTool registers a tool whose handler returns a raw *mcp.CallToolResult.
// Use this when you need to return multiple content items or non-text content.
// Must be called before Start.
func (f *FakeMCPServer) AddRawTool(name, description string, fn func(ctx context.Context, args map[string]any) (*mcp.CallToolResult, error)) {
	f.t.Helper()
	inputSchema := json.RawMessage(`{"type":"object"}`)
	f.server.AddTool(
		&mcp.Tool{
			Name:        name,
			Description: description,
			InputSchema: inputSchema,
		},
		func(ctx context.Context, req *mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			var args map[string]any
			if len(req.Params.Arguments) > 0 {
				if err := json.Unmarshal(req.Params.Arguments, &args); err != nil {
					return nil, err
				}
			}
			return fn(ctx, args)
		},
	)
}

// Start connects the server and client in-process and registers cleanup.
// Must be called before Client().
func (f *FakeMCPServer) Start() {
	f.t.Helper()
	ctx := context.Background()

	clientTransport, serverTransport := mcp.NewInMemoryTransports()

	// Connect server side.
	serverSession, err := f.server.Connect(ctx, serverTransport, nil)
	if err != nil {
		f.t.Fatalf("FakeMCPServer: server connect: %v", err)
	}
	f.t.Cleanup(func() { serverSession.Close() })

	// Connect client side.
	c := mcp.NewClient(&mcp.Implementation{Name: "fake-mcp-client", Version: "v0.0.1"}, nil)
	clientSession, err := c.Connect(ctx, clientTransport, nil)
	if err != nil {
		f.t.Fatalf("FakeMCPServer: client connect: %v", err)
	}
	f.t.Cleanup(func() { clientSession.Close() })

	f.client = clientSession
}

// Client returns the connected MCP client session.
// Panics if Start has not been called.
func (f *FakeMCPServer) Client() *mcp.ClientSession {
	if f.client == nil {
		panic("FakeMCPServer: Start has not been called")
	}
	return f.client
}

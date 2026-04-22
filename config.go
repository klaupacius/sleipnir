package sleipnir

import (
	"fmt"
	"log/slog"
	"os"
	"strconv"
	"time"
)

const (
	defaultMaxIterations    = 20
	defaultMaxParallelTools = 8
	defaultCompactThreshold = 0.75
	defaultCompactModel     = "claude-haiku-4-5"
	defaultMaxLLMRetries    = 3
	defaultHITLTimeout      = 30 * time.Minute
	defaultLogLevel         = slog.LevelInfo
	defaultLogFormat        = "text"
)

// Config holds harness-wide settings. All fields are optional; zero values
// are replaced with documented defaults by [resolveDefaults].
//
// Note on [Config.Middlewares]: nil and empty slice are both valid; the
// harness treats them identically.
type Config struct {
	DefaultMaxIterations    int
	DefaultMaxParallelTools int
	CompactThreshold        float64 // 0.75 = 75% of model context; validated (0.0, 1.0]
	CompactModel            string
	MaxLLMRetries           int
	HITLTimeout             time.Duration
	// LogLevel controls the minimum log level for the harness logger.
	//
	// NOTE: this field is only honoured when set via [LoadConfigFromEnv] or
	// the SLEIPNIR_LOG_LEVEL environment variable. Direct struct construction
	// cannot set the internal logLevelSet flag, so [resolveDefaults] will
	// silently override any value set here to [slog.LevelInfo].
	LogLevel              slog.Level
	LogFormat             string // "json" | "text"
	Middlewares           []Middleware
	AllowLateRegistration bool

	logLevelSet bool
}

// LoadConfigFromEnv reads SLEIPNIR_* environment variables and returns a
// fully resolved Config. Missing variables fall back to documented defaults.
// No files are read; no globals are mutated.
func LoadConfigFromEnv() (Config, error) {
	var cfg Config
	var err error

	if s, ok := lookupEnv("SLEIPNIR_MAX_ITERATIONS"); ok {
		cfg.DefaultMaxIterations, err = parseInt(s, "SLEIPNIR_MAX_ITERATIONS")
		if err != nil {
			return Config{}, err
		}
	}

	if s, ok := lookupEnv("SLEIPNIR_MAX_PARALLEL_TOOLS"); ok {
		cfg.DefaultMaxParallelTools, err = parseInt(s, "SLEIPNIR_MAX_PARALLEL_TOOLS")
		if err != nil {
			return Config{}, err
		}
	}

	if s, ok := lookupEnv("SLEIPNIR_COMPACT_THRESHOLD"); ok {
		cfg.CompactThreshold, err = parseFloat(s, "SLEIPNIR_COMPACT_THRESHOLD")
		if err != nil {
			return Config{}, err
		}
	}

	if s, ok := lookupEnv("SLEIPNIR_COMPACT_MODEL"); ok {
		cfg.CompactModel = s
	}

	if s, ok := lookupEnv("SLEIPNIR_MAX_LLM_RETRIES"); ok {
		cfg.MaxLLMRetries, err = parseInt(s, "SLEIPNIR_MAX_LLM_RETRIES")
		if err != nil {
			return Config{}, err
		}
	}

	if s, ok := lookupEnv("SLEIPNIR_HITL_TIMEOUT"); ok {
		cfg.HITLTimeout, err = time.ParseDuration(s)
		if err != nil {
			return Config{}, fmt.Errorf("sleipnir: SLEIPNIR_HITL_TIMEOUT: invalid duration %q: %w", s, err)
		}
	}

	if s, ok := lookupEnv("SLEIPNIR_LOG_LEVEL"); ok {
		cfg.LogLevel, err = parseLogLevel(s)
		if err != nil {
			return Config{}, err
		}
		cfg.logLevelSet = true
	}

	if s, ok := lookupEnv("SLEIPNIR_LOG_FORMAT"); ok {
		if s != "json" && s != "text" {
			return Config{}, fmt.Errorf("sleipnir: SLEIPNIR_LOG_FORMAT: must be \"json\" or \"text\", got %q", s)
		}
		cfg.LogFormat = s
	}

	if s, ok := lookupEnv("SLEIPNIR_ALLOW_LATE_REGISTRATION"); ok {
		cfg.AllowLateRegistration, err = strconv.ParseBool(s)
		if err != nil {
			return Config{}, fmt.Errorf("sleipnir: SLEIPNIR_ALLOW_LATE_REGISTRATION: must be \"true\" or \"false\", got %q", s)
		}
	}

	return resolveDefaults(cfg), nil
}

// lookupEnv wraps os.LookupEnv and trims empty strings so that set-but-empty
// vars are treated as absent.
func lookupEnv(key string) (string, bool) {
	s, ok := os.LookupEnv(key)
	if !ok || s == "" {
		return "", false
	}
	return s, true
}

func parseInt(s, envKey string) (int, error) {
	v, err := strconv.Atoi(s)
	if err != nil {
		return 0, fmt.Errorf("sleipnir: %s: invalid integer %q: %w", envKey, s, err)
	}
	return v, nil
}

func parseFloat(s, envKey string) (float64, error) {
	v, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return 0, fmt.Errorf("sleipnir: %s: invalid float %q: %w", envKey, s, err)
	}
	return v, nil
}

func parseLogLevel(s string) (slog.Level, error) {
	switch s {
	case "DEBUG", "debug":
		return slog.LevelDebug, nil
	case "INFO", "info":
		return slog.LevelInfo, nil
	case "WARN", "warn", "WARNING", "warning":
		return slog.LevelWarn, nil
	case "ERROR", "error":
		return slog.LevelError, nil
	default:
		return 0, fmt.Errorf("sleipnir: SLEIPNIR_LOG_LEVEL: must be DEBUG|INFO|WARN|ERROR, got %q", s)
	}
}

func resolveDefaults(cfg Config) Config {
	if cfg.DefaultMaxIterations == 0 {
		cfg.DefaultMaxIterations = defaultMaxIterations
	}
	if cfg.DefaultMaxParallelTools == 0 {
		cfg.DefaultMaxParallelTools = defaultMaxParallelTools
	}
	if cfg.CompactThreshold == 0.0 {
		cfg.CompactThreshold = defaultCompactThreshold
	}
	if cfg.CompactModel == "" {
		cfg.CompactModel = defaultCompactModel
	}
	if cfg.MaxLLMRetries == 0 {
		cfg.MaxLLMRetries = defaultMaxLLMRetries
	}
	if cfg.HITLTimeout == 0 {
		cfg.HITLTimeout = defaultHITLTimeout
	}
	if !cfg.logLevelSet {
		cfg.LogLevel = defaultLogLevel
	}
	if cfg.LogFormat == "" {
		cfg.LogFormat = defaultLogFormat
	}
	// Middlewares: nil is intentionally left as-is.
	return cfg
}

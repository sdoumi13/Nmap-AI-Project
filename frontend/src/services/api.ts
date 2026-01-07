// src/services/api.ts - FIXED VERSION

const API_BASE_URL = 'http://localhost:8001';
const HISTORY_URL = 'http://localhost:5002'; // Agent 5 MCP Server - UPDATED PORT
const HEALTH_URL = 'http://localhost:5002'; // Health endpoint - UPDATED PORT

// ============================================================================
// TYPE DEFINITIONS
// ============================================================================

export interface HistoryEntry {
  id: string;
  query: string;
  timestamp: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  target_agent: string | null;
  generated_command: string | null;
  generation_method: string | null;
  execution_report: any | null;
}

export interface ComplexityAnalysis {
  level: 'Easy' | 'Medium' | 'Hard';
  confidence: number;
  reason: string;
  target_agent: string;
}

export interface AnalyzeResponse {
  status: string;
  relevant: boolean;
  reason?: string;
  analysis: ComplexityAnalysis;
  generated_command?: string;
  best_match_command?: string;
  generation_method?: string;
  entry_id?: string;
}

export interface ValidationResult {
  valid: boolean;
  score: number;
  status: string;
  method: string;
  errors: string[];
  warnings: string[];
}

export interface ExecutionStages {
  validation?: {
    status: string;
    score: number;
    method: string;
    errors?: string[];
  };
  self_correction?: {
    applied: boolean;
    original_command?: string;
    corrected_command?: string;
    final_command?: string;
    final_score?: number;
    attempts?: any[];
    history?: any[];
  };
  sandbox_execution?: {
    success: boolean;
    command: string;
    exit_code: number;
    output: string;
    runtime: number;
  };
  sandbox?: {
    success: boolean;
    command: string;
    exit_code: number;
    output: string;
    runtime: number;
  };
  vm_execution?: {
    success: boolean;
    command: string;
    target: string;
    exit_code: number;
    output: string;
    runtime: number;
  };
}

export interface ExecutionReport {
  final_status: string;
  command: string;
  timestamp: string;
  stages: ExecutionStages;
  report?: ExecutionStages; // Alias for compatibility
}

export interface HealthStatus {
  status: 'online' | 'offline';
  local_agents: {
    comprehension_ready: boolean;
    rag_ready: boolean;
    diffusion_ready: boolean;
  };
  external_services: {
    complexity_api?: {
      status: string;
      url: string;
    };
    agent5_mcp?: {
      status: string;
      url: string;
    };
  };
}

// ============================================================================
// API CLIENTS
// ============================================================================

/**
 * Analyze API - Routes user query through Router Agent
 * FIXED: This is now a direct function, not an object with .analyze method
 */
export const analyzeApi = async (params: { 
  query: string; 
  target: string;
}): Promise<AnalyzeResponse> => {
  try {
    const response = await fetch(`${API_BASE_URL}/route`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        query: params.query,
        target: params.target,
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Router API error (${response.status}): ${errorText}`);
    }

    const data = await response.json();
    
    // Transform the response to match frontend expectations
    return {
      status: data.status || 'unknown',
      relevant: data.status !== 'rejected',
      reason: data.reason,
      analysis: data.complexity || {
        level: 'Medium',
        confidence: 0,
        reason: 'Unknown',
        target_agent: data.agent || 'UNKNOWN'
      },
      generated_command: data.command_generated,
      generation_method: data.agent,
      entry_id: data.execution?.entry_id || `entry_${Date.now()}`,
    };
  } catch (error) {
    console.error('Analyze API Error:', error);
    throw error;
  }
};

/**
 * History API - Fetches command execution history
 */
export const historyApi = {
  async getAll(): Promise<HistoryEntry[]> {
    try {
      const response = await fetch(`${HISTORY_URL}/history`);
      
      if (!response.ok) {
        console.warn(`History API returned ${response.status}, returning empty array`);
        return [];
      }

      const data = await response.json();
      return Array.isArray(data) ? data : [];
    } catch (error) {
      console.error('History API Error:', error);
      return []; // Return empty array instead of throwing
    }
  },

  async getById(id: string): Promise<HistoryEntry | null> {
    try {
      const response = await fetch(`${HISTORY_URL}/history/${id}`);
      
      if (!response.ok) {
        throw new Error(`Failed to fetch entry ${id}`);
      }

      return await response.json();
    } catch (error) {
      console.error('History Get By ID Error:', error);
      return null;
    }
  }
};

/**
 * Health API - Checks system health status
 */
export const healthApi = {
  async check(): Promise<HealthStatus> {
    try {
      const response = await fetch(`${HEALTH_URL}/health`);
      
      if (!response.ok) {
        throw new Error(`Health check failed with status ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Health API Error:', error);
      // Return offline status on error
      return {
        status: 'offline',
        local_agents: {
          comprehension_ready: false,
          rag_ready: false,
          diffusion_ready: false,
        },
        external_services: {}
      };
    }
  }
};

/**
 * Validation API - Validates commands
 */
export const validationApi = {
  async validate(params: {
    entry_id: string;
    intent: string;
    command: string;
    agent_name: string;
  }): Promise<ValidationResult> {
    try {
      const response = await fetch(`${HISTORY_URL}/mcp/validate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          command: params.command,
          intent: params.intent,
          agent_name: params.agent_name,
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Validation failed (${response.status}): ${errorText}`);
      }

      const data = await response.json();
      
      return {
        valid: data.valid || false,
        score: data.score || 0,
        status: data.status || 'unknown',
        method: data.method_used || 'unknown',
        errors: data.errors || [],
        warnings: data.warnings || [],
      };
    } catch (error) {
      console.error('Validation API Error:', error);
      throw error;
    }
  }
};

/**
 * Execution API - Executes validated commands
 */
export const executionApi = {
  async execute(params: {
    entry_id: string;
    intent: string;
    command: string;
    target: string;
    agent_name: string;
  }): Promise<ExecutionReport> {
    try {
      const response = await fetch(`${HISTORY_URL}/mcp/execute`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          command: params.command,
          intent: params.intent,
          target: params.target,
          agent_name: params.agent_name,
          skip_sandbox: false,
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Execution failed (${response.status}): ${errorText}`);
      }

      const data = await response.json();
      
      return {
        final_status: data.final_status || 'unknown',
        command: data.command || params.command,
        timestamp: data.timestamp || new Date().toISOString(),
        stages: data.stages || data.report || {},
        report: data.report || data.stages || {},
      };
    } catch (error) {
      console.error('Execution API Error:', error);
      throw error;
    }
  }
};

/**
 * Helper function to check if API is reachable
 */
export const checkApiConnection = async (): Promise<boolean> => {
  try {
    const response = await fetch(`${HEALTH_URL}/health`, {
      method: 'GET',
      signal: AbortSignal.timeout(5000), // 5 second timeout
    });
    return response.ok;
  } catch (error) {
    console.error('API Connection Check Failed:', error);
    return false;
  }
};

export default {
  analyze: analyzeApi,
  history: historyApi,
  health: healthApi,
  validation: validationApi,
  execution: executionApi,
  checkConnection: checkApiConnection,
};
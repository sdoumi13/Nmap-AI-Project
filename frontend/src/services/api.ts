const API_BASE_URL = 'http://localhost:8000/api';

// --- INTERFACES POUR LE RAPPORT D'EXÉCUTION (AGENT 5) ---

export interface ExecutionStage {
  success?: boolean;
  status?: string;
  output?: string;
  errors?: string[];
  time?: number;
  [key: string]: string | number | boolean | object | undefined; // Pour la flexibilité des métadonnées
}

export interface ExecutionReport {
  final_status: 'success' | 'failed_validation' | 'failed_sandbox' | 'failed_vm' | 'vm_connection_error';
  intent: string;
  original_command: string;
  target: string;
  agent: string;
  timestamp: string;
  stages: {
    validation?: {
      status: string;
      score: number;
      method: string;
      errors?: string[];
    };
    self_correction?: {
      applied: boolean;
      final_command?: string;
      history?: string[];
    };
    sandbox?: ExecutionStage;
    vm_execution?: ExecutionStage & { exit_code?: number };
  };
}

// --- INTERFACE PRINCIPALE DE L'HISTORIQUE ---

export interface HistoryEntry {
  id: string;
  query: string;
  best_match_command?: string;
  generated_command?: string;
  generation_method?: string;
  complexity: string;
  status: 'completed' | 'failed' | 'pending' | 'running';
  timestamp: string;
  target_agent?: string;
  execution_report?: ExecutionReport | null; // Contiendra tout le détail de l'Agent 5
}

// --- RÉPONSES DES ENDPOINTS ---

export interface AnalyzeResponse {
  relevant: boolean;
  entry_id: string; // Crucial pour l'étape d'exécution
  best_match_command?: string;
  generated_command?: string;
  generation_method?: string;
  analysis: {
    level: string;
    target_agent: string;
    confidence?: number;
    reason: string;
  };
  status?: string;
  reason?: string;
}

// --- SERVICES API ---

/**
 * Agent 1: Analyse de la requête et détermination de la complexité
 */
export const analyzeApi = {
  analyze: async (query: string, target?: string): Promise<AnalyzeResponse> => {
    try {
      const response = await fetch(`${API_BASE_URL}/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          query,
          target: target || '192.168.188.128'
        }),
      });
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || errorData.message || `Erreur HTTP ${response.status}: Erreur lors de l'analyse IA`);
      }
      
      return await response.json();
    } catch (error) {
      if (error instanceof TypeError && error.message.includes('fetch')) {
        throw new Error('Erreur de connexion: Impossible d\'atteindre le serveur API');
      }
      throw error;
    }
  }
};

/**
 * Agent 5: Validation seule d'une commande
 */
export const validationApi = {
  validate: async (data: {
    entry_id: string;
    intent: string;
    command: string;
    agent_name: string;
  }): Promise<{
    valid: boolean;
    status: string;
    score: number;
    errors: string[];
    warnings: string[];
    method_used: string;
    timestamp: string;
  }> => {
    try {
      const response = await fetch(`${API_BASE_URL}/validate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          entry_id: data.entry_id,
          intent: data.intent,
          command: data.command,
          agent_name: data.agent_name,
          target: '192.168.188.128' // Required by backend but not used for validation
        }),
      });
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || errorData.message || `Erreur HTTP ${response.status}: Échec de la validation`);
      }
      
      return await response.json();
    } catch (error) {
      if (error instanceof TypeError && error.message.includes('fetch')) {
        throw new Error('Erreur de connexion: Impossible d\'atteindre le serveur API');
      }
      throw error;
    }
  }
};

/**
 * Agent 5: Validation, Sandbox et Exécution finale sur VM
 */
export const executionApi = {
  execute: async (data: {
    entry_id: string;
    intent: string;
    command: string;
    target: string;
    agent_name: string;
  }): Promise<ExecutionReport> => {
    try {
      const response = await fetch(`${API_BASE_URL}/execute`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
      });
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || errorData.message || `Erreur HTTP ${response.status}: Échec du pipeline d'exécution`);
      }
      
      return await response.json();
    } catch (error) {
      if (error instanceof TypeError && error.message.includes('fetch')) {
        throw new Error('Erreur de connexion: Impossible d\'atteindre le serveur API');
      }
      throw error;
    }
  }
};

/**
 * Persistance: Récupération de l'historique et des stats
 */
export const historyApi = {
  getAll: async (): Promise<HistoryEntry[]> => {
    try {
      const response = await fetch(`${API_BASE_URL}/history`, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
      });
      
      if (!response.ok) {
        throw new Error(`Erreur HTTP: ${response.status} - Impossible de charger l'historique`);
      }
      
      const data = await response.json();
      
      // Vérifier que la réponse est un tableau
      if (Array.isArray(data)) {
        return data;
      }
      
      // Si l'API retourne un objet avec une clé 'history' ou similaire
      if (data && Array.isArray(data.history)) {
        return data.history;
      }
      
      // Si l'API retourne un objet avec une clé 'data'
      if (data && Array.isArray(data.data)) {
        return data.data;
      }
      
      // Par défaut, retourner un tableau vide si la structure est inattendue
      console.warn('Format de réponse inattendu de l\'API /history:', data);
      return [];
    } catch (error) {
      if (error instanceof TypeError && error.message.includes('fetch')) {
        throw new Error('Erreur de connexion: Impossible d\'atteindre le serveur API');
      }
      throw error;
    }
  }
};

/**
 * Génération de commande: Endpoint dédié pour STEP 3
 */
export const generateApi = {
  generate: async (data: {
    query: string;
    target?: string;
    agent_type?: 'RAG' | 'DIFFUSION';
  }): Promise<{ command: string; agent_type: string; query: string; target: string }> => {
    try {
      const response = await fetch(`${API_BASE_URL}/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: data.query,
          target: data.target || '192.168.188.128',
          agent_type: data.agent_type || undefined,
        }),
      });
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || errorData.message || `Erreur HTTP ${response.status}: Échec de la génération de commande`);
      }
      
      return await response.json();
    } catch (error) {
      if (error instanceof TypeError && error.message.includes('fetch')) {
        throw new Error('Erreur de connexion: Impossible d\'atteindre le serveur API');
      }
      throw error;
    }
  }
};

// --- INTERFACE POUR LE HEALTH CHECK ---

export interface ExternalServiceStatus {
  status: 'online' | 'offline' | 'error';
  url: string;
  error?: string;
}

export interface HealthStatus {
  status: string;
  local_agents: {
    comprehension_ready: boolean;
    rag_ready: boolean;
    diffusion_ready: boolean;
  };
  external_services: {
    complexity_api?: ExternalServiceStatus;
    agent5_mcp?: ExternalServiceStatus;
  };
}

/**
 * Utilitaires: Vérification de l'état des agents et services
 */
export const healthApi = {
  check: async (): Promise<HealthStatus> => {
    try {
      // Créer un AbortController pour gérer le timeout
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5000); // 5 secondes
      
      const response = await fetch(`${API_BASE_URL}/health`, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
        signal: controller.signal,
      });
      
      clearTimeout(timeoutId);
      
      if (!response.ok) {
        throw new Error(`Health check failed: ${response.status}`);
      }
      
      const data = await response.json();
      return data as HealthStatus;
    } catch (error) {
      // Si c'est une erreur d'abort (timeout), on la propage
      if (error instanceof Error && error.name === 'AbortError') {
        throw new Error('Timeout: Le serveur ne répond pas');
      }
      // Si c'est une erreur réseau (CORS, connexion, etc.)
      if (error instanceof TypeError && error.message.includes('fetch')) {
        throw new Error('Erreur de connexion: Impossible d\'atteindre le serveur API');
      }
      // Sinon, on propage l'erreur
      throw error;
    }
  }
};
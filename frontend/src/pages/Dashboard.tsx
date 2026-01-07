import { useState, useEffect, useCallback, memo } from 'react';
import { Link } from 'react-router-dom';
import { 
  Network, ShieldCheck, Play, Clock, CheckCircle, XCircle, Activity, 
  ArrowRight, TrendingUp, AlertTriangle, Zap, Globe, Server, 
  Cpu, HardDrive, Wifi, Lock, Code, Sparkles, Brain, Terminal
} from 'lucide-react';
import { historyApi, healthApi, validationApi, executionApi, HistoryEntry, HealthStatus } from '../services/api';
import RobotPipeline from '../components/RobotPipeline';



export default function Dashboard() {
  const [stats, setStats] = useState({ total: 0, completed: 0, failed: 0, pending: 0, running: 0 });
  const [recentHistory, setRecentHistory] = useState<HistoryEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [apiHealth, setApiHealth] = useState<boolean>(false);
  const [healthStatus, setHealthStatus] = useState<HealthStatus | null>(null);
  const [actionLoading, setActionLoading] = useState<'validation' | 'execution' | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [actionSuccess, setActionSuccess] = useState<string | null>(null);

  const fetchDashboardData = useCallback(async () => {
    // Note: On ne met pas setLoading(true) ici pour éviter le clignotement lors du refresh auto
    try {
      try {
        const health = await healthApi.check();
        setHealthStatus(health);
        setApiHealth(health.status === 'online');
      } catch {
        setApiHealth(false);
        setHealthStatus(null);
      }

      const data: HistoryEntry[] = await historyApi.getAll();

      // Calculer les stats de manière optimisée (une seule passe)
      let completed = 0, failed = 0, pending = 0, running = 0;
      for (const h of data) {
        if (h.status === 'completed') completed++;
        else if (h.status === 'failed') failed++;
        else if (h.status === 'pending') pending++;
        else if (h.status === 'running') running++;
      }
      
      setStats({
        total: data.length,
        completed,
        failed,
        pending,
        running,
      });
      
      // Trier et limiter une seule fois
      const sortedData = [...data].sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());
      setRecentHistory(sortedData.slice(0, 5));

    } catch (error) {
      console.error('Erreur Dashboard:', error);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchDashboardData();
    // Augmenter l'intervalle à 60s pour réduire la charge
    const interval = setInterval(fetchDashboardData, 60000);
    return () => clearInterval(interval);
  }, [fetchDashboardData]);

  const getStatusColor = useCallback((status: HistoryEntry['status']) => {
    switch (status) {
        case 'completed': return 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20';
        case 'failed': return 'bg-red-500/10 text-red-400 border border-red-500/20';
        case 'running': return 'bg-blue-500/10 text-blue-400 border border-blue-500/20 animate-pulse';
        default: return 'bg-yellow-500/10 text-yellow-400 border border-yellow-500/20';
    }
  }, []);

  // Trouve la dernière entrée en attente pour validation
  const getLastPendingEntry = useCallback((): HistoryEntry | null => {
    const pending = recentHistory.filter(h => h.status === 'pending' && h.generated_command);
    if (pending.length === 0) {
      // Chercher dans tout l'historique si nécessaire
      return null;
    }
    return pending[0]; // La plus récente (déjà triée)
  }, [recentHistory]);

  // Trouve la dernière entrée validée pour exécution
  const getLastValidatedEntry = useCallback((): HistoryEntry | null => {
    // Chercher une entrée qui a une commande mais pas encore d'exécution
    const candidates = recentHistory.filter(h => 
      h.generated_command && 
      h.status === 'pending' && 
      !h.execution_report
    );
    return candidates.length > 0 ? candidates[0] : null;
  }, [recentHistory]);

  const handleQuickValidation = useCallback(async () => {
    const entry = getLastPendingEntry();
    if (!entry || !entry.generated_command) {
      setActionError('Aucune commande en attente de validation');
      setTimeout(() => setActionError(null), 3000);
      return;
    }

    setActionLoading('validation');
    setActionError(null);
    setActionSuccess(null);

    try {
      const result = await validationApi.validate({
        entry_id: entry.id,
        intent: entry.query,
        command: entry.generated_command,
        agent_name: entry.target_agent || 'UNKNOWN'
      });

      if (result.valid) {
        setActionSuccess(`Validation réussie ! Score: ${result.score}/100`);
        // Rafraîchir les données
        setTimeout(() => {
          fetchDashboardData();
          setActionSuccess(null);
        }, 2000);
      } else {
        setActionError(`Validation échouée. Score: ${result.score}/100. Erreurs: ${result.errors.join(', ')}`);
      }
    } catch (err) {
      const errorMessage = (err instanceof Error) ? err.message : 'Erreur lors de la validation';
      setActionError(errorMessage);
    } finally {
      setActionLoading(null);
      setTimeout(() => {
        setActionError(null);
        setActionSuccess(null);
      }, 5000);
    }
  }, [getLastPendingEntry, fetchDashboardData]);

  const handleQuickExecution = useCallback(async () => {
    const entry = getLastValidatedEntry();
    if (!entry || !entry.generated_command) {
      setActionError('Aucune commande prête pour l\'exécution');
      setTimeout(() => setActionError(null), 3000);
      return;
    }

    setActionLoading('execution');
    setActionError(null);
    setActionSuccess(null);

    try {
      const result = await executionApi.execute({
        entry_id: entry.id,
        intent: entry.query,
        command: entry.generated_command,
        target: '192.168.188.128', // Default target
        agent_name: entry.target_agent || 'UNKNOWN'
      });

      if (result.final_status === 'success') {
        setActionSuccess('Exécution réussie sur la VM !');
        // Rafraîchir les données
        setTimeout(() => {
          fetchDashboardData();
          setActionSuccess(null);
        }, 2000);
      } else {
        setActionError(`Exécution échouée: ${result.final_status}`);
      }
    } catch (err) {
      const errorMessage = (err instanceof Error) ? err.message : 'Erreur lors de l\'exécution';
      setActionError(errorMessage);
    } finally {
      setActionLoading(null);
      setTimeout(() => {
        setActionError(null);
        setActionSuccess(null);
      }, 5000);
    }
  }, [getLastValidatedEntry, fetchDashboardData]);

  // Determine current pipeline stage based on recent history
  const getCurrentPipelineStage = useCallback((): number => {
    if (!recentHistory.length) return 0;
    const latest = recentHistory[0];
    if (latest.status === 'running') {
      // Determine which stage based on what data is available
      if (latest.generated_command && latest.execution_report) return 4;
      if (latest.generated_command) return 3;
      if (latest.target_agent) return 2;
      return 1;
    }
    if (latest.status === 'completed') return 4;
    if (latest.status === 'pending') return latest.generated_command ? 3 : 2;
    return 0;
  }, [recentHistory]);

  // Get robot statuses based on health and history
  const getRobotStatuses = useCallback(() => {
    const statuses: Record<string, 'idle' | 'active' | 'processing' | 'complete' | 'error'> = {};
    const logs: Record<string, string[]> = {};
    
    if (healthStatus) {
      statuses.comprehension = healthStatus.local_agents.comprehension_ready ? 'active' : 'idle';
      logs.comprehension = ['TF-IDF Analysis Ready', 'SBERT Embeddings Loaded'];
      
      statuses.router = healthStatus.external_services.complexity_api?.status === 'online' ? 'active' : 'idle';
      logs.router = ['Complexity API: Connected', 'Routing Engine: Ready'];
      
      statuses.agent5 = healthStatus.external_services.agent5_mcp?.status === 'online' ? 'active' : 'idle';
      logs.agent5 = ['MCP Server: Active', 'Validation: Ready', 'Sandbox: Docker', 'VM: Connected'];
    }

    // Determine active model from latest history entry
    if (recentHistory.length > 0) {
      const latest = recentHistory[0];
      const method = latest.generation_method?.toLowerCase() || '';
      if (method.includes('rag') || method.includes('mistral')) {
        statuses.mistral = latest.status === 'running' ? 'processing' : latest.status === 'completed' ? 'complete' : 'active';
      } else if (method.includes('phi')) {
        statuses.phi4 = latest.status === 'running' ? 'processing' : latest.status === 'completed' ? 'complete' : 'active';
      } else if (method.includes('lora') || method.includes('t5')) {
        statuses['lora-t5'] = latest.status === 'running' ? 'processing' : latest.status === 'completed' ? 'complete' : 'active';
      }
    }

    return { statuses, logs };
  }, [healthStatus, recentHistory]);

  const { statuses, logs } = getRobotStatuses();
  const currentStage = getCurrentPipelineStage();
  const latestQuery = recentHistory.length > 0 ? recentHistory[0].query : '';

  return (
    <div className="relative min-h-screen bg-claude-charcoal-dark text-claude-white">
      {/* Claude Aesthetic Background */}
      <div className="absolute inset-0 -z-10 overflow-hidden pointer-events-none">
        <div className="absolute top-0 right-0 w-96 h-96 bg-claude-coral/5 rounded-full blur-3xl" />
        <div className="absolute bottom-0 left-0 w-[500px] h-[500px] bg-claude-white/3 rounded-full blur-3xl" />
        {/* Subtle grid pattern */}
        <div 
          className="absolute inset-0 opacity-5"
          style={{
            backgroundImage: `
              linear-gradient(rgba(255, 255, 255, 0.1) 1px, transparent 1px),
              linear-gradient(90deg, rgba(255, 255, 255, 0.1) 1px, transparent 1px)
            `,
            backgroundSize: '30px 30px',
          }}
        />
      </div>

      <div className="space-y-8 relative z-10 p-6">
        {/* Header Section - Claude Style */}
        <div className="flex flex-col md:flex-row justify-between items-start md:items-end gap-4 border-b border-claude-white/10 pb-6 relative animate-fade-in-up">
          <div className="relative">
            <h1 className="text-5xl md:text-6xl font-bold text-claude-white tracking-tight relative">
              <span className="bg-clip-text text-transparent bg-gradient-to-r from-claude-white via-claude-coral to-claude-white">
                Dashboard
              </span>
            </h1>
            <p className="mt-3 text-claude-grey-light text-lg flex items-center gap-2">
              <Server className="w-4 h-4 text-claude-coral" />
              Surveillance et orchestration des agents IA
            </p>
          </div>
          
          <div 
            className={`flex items-center gap-3 px-5 py-3 rounded-full border transition-all duration-300 shadow-lg ${
              apiHealth 
                ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400 shadow-emerald-500/20' 
                : 'bg-red-500/10 border-red-500/30 text-red-400 shadow-red-500/20'
            }`}
          >
            <div 
              className={`w-3 h-3 rounded-full ${apiHealth ? 'bg-emerald-500 animate-pulse' : 'bg-red-500'}`}
            />
            <span className="font-mono text-sm font-semibold tracking-wider">
              API {apiHealth ? 'ONLINE' : 'OFFLINE'}
            </span>
          </div>
        </div>

        {/* Robot Pipeline Visualization - Main Feature */}
        <div className="animate-fade-in-up animate-stagger-1">
          <RobotPipeline 
            currentStage={currentStage}
            query={latestQuery}
            status={statuses}
            logs={logs}
            healthStatus={healthStatus}
          />
        </div>

        {/* Services Status Section - Claude Style */}
        {healthStatus && (
          <div className="bg-gradient-to-br from-claude-charcoal/90 via-claude-charcoal/80 to-claude-charcoal-dark/90 border border-claude-white/10 rounded-2xl p-6 animate-fade-in-up animate-stagger-1 backdrop-blur-sm"
          >
            <h3 className="text-lg font-bold text-claude-white mb-4 flex items-center gap-2">
              <Server className="w-5 h-5 text-claude-coral" />
              Statut des Services
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Local Agents */}
              <div className="space-y-2">
                <h4 className="text-sm font-semibold text-claude-grey-light uppercase tracking-wider">Agents Locaux</h4>
                <ServiceStatusItem 
                  label="Comprehension Agent" 
                  ready={healthStatus.local_agents.comprehension_ready}
                  icon={Brain}
                />
                <ServiceStatusItem 
                  label="RAG Agent" 
                  ready={healthStatus.local_agents.rag_ready}
                  icon={Sparkles}
                />
                <ServiceStatusItem 
                  label="Diffusion Model" 
                  ready={healthStatus.local_agents.diffusion_ready}
                  icon={Code}
                />
              </div>
              
              {/* External Services */}
              <div className="space-y-2">
                <h4 className="text-sm font-semibold text-claude-grey-light uppercase tracking-wider">Services Externes</h4>
                {healthStatus.external_services.complexity_api && (
                  <ServiceStatusItem 
                    label="Complexity API (Port 7000)" 
                    ready={healthStatus.external_services.complexity_api.status === 'online'}
                    icon={Network}
                  />
                )}
                {healthStatus.external_services.agent5_mcp && (
                  <ServiceStatusItem 
                    label="Agent 5 MCP (Port 5000)" 
                    ready={healthStatus.external_services.agent5_mcp.status === 'online'}
                    icon={Terminal}
                  />
                )}
              </div>
            </div>
          </div>
        )}

        {/* Stats Grid Enhanced - Claude Style */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <div className="animate-fade-in-up animate-stagger-1">
            <StatCard 
              title="Total Requêtes" 
              value={stats.total} 
              icon={Network} 
              color="text-claude-coral" 
              bgGradient="from-claude-coral/20 via-claude-coral/10 to-transparent"
              borderColor="border-claude-coral/30"
              loading={loading}
              trend={stats.total > 0 ? 'up' : 'neutral'}
            />
          </div>
          <div className="animate-fade-in-up animate-stagger-2">
            <StatCard 
              title="Succès" 
              value={stats.completed} 
              icon={CheckCircle} 
              color="text-emerald-400" 
              bgGradient="from-emerald-500/20 via-emerald-500/10 to-transparent"
              borderColor="border-emerald-500/30"
              loading={loading}
              trend={stats.completed > 0 ? 'up' : 'neutral'}
            />
          </div>
          <div className="animate-fade-in-up animate-stagger-3">
            <StatCard 
              title="Échecs" 
              value={stats.failed} 
              icon={XCircle} 
              color="text-red-400" 
              bgGradient="from-red-500/20 via-red-500/10 to-transparent"
              borderColor="border-red-500/30"
              loading={loading}
              trend={stats.failed > 0 ? 'down' : 'neutral'}
            />
          </div>
          <div className="animate-fade-in-up animate-stagger-4">
            <StatCard 
              title="Actifs" 
              value={stats.pending + stats.running} 
              icon={Activity} 
              color="text-claude-coral" 
              bgGradient="from-claude-coral/20 via-claude-coral/10 to-transparent"
              borderColor="border-claude-coral/30"
              loading={loading}
              trend={stats.pending + stats.running > 0 ? 'up' : 'neutral'}
            />
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Quick Actions Enhanced - Claude Style */}
          <div className="lg:col-span-1 space-y-6 animate-fade-in-up animate-stagger-2">
            <h2 className="text-2xl font-bold text-claude-white flex items-center gap-3">
              <Zap className="w-6 h-6 text-claude-coral" />
              Actions Rapides
            </h2>
            <div className="grid gap-4">
              <ActionCard 
                to="/router" 
                icon={Network} 
                title="Nouvelle Analyse" 
                subtitle="Router IA" 
                color="blue"
                description="Lancer une nouvelle analyse réseau"
              />
              <QuickActionCard 
                onClick={handleQuickValidation}
                icon={ShieldCheck} 
                title="Validation Rapide" 
                subtitle="Security Check" 
                color="emerald"
                description={getLastPendingEntry() ? `Valider: ${getLastPendingEntry()?.generated_command?.substring(0, 40)}...` : "Aucune commande en attente"}
                loading={actionLoading === 'validation'}
                disabled={!getLastPendingEntry() || !getLastPendingEntry()?.generated_command}
              />
              <QuickActionCard 
                onClick={handleQuickExecution}
                icon={Play} 
                title="Exécution Rapide" 
                subtitle="Sandbox & VM" 
                color="purple"
                description={getLastValidatedEntry() ? `Exécuter: ${getLastValidatedEntry()?.generated_command?.substring(0, 40)}...` : "Aucune commande prête"}
                loading={actionLoading === 'execution'}
                disabled={!getLastValidatedEntry() || !getLastValidatedEntry()?.generated_command}
              />
            </div>

            {/* System Status Cards */}
            <div className="mt-6 space-y-3">
              <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-2">
                <Globe className="w-4 h-4" />
                Statut Système
              </h3>
              <div className="grid grid-cols-2 gap-3">
                <SystemStatusCard 
                  icon={Cpu}
                  label="CPU"
                  value="45%"
                  status="normal"
                />
                <SystemStatusCard 
                  icon={HardDrive}
                  label="Storage"
                  value="62%"
                  status="normal"
                />
                <SystemStatusCard 
                  icon={Wifi}
                  label="Network"
                  value="Active"
                  status="active"
                />
                <SystemStatusCard 
                  icon={Lock}
                  label="Security"
                  value="Secure"
                  status="secure"
                />
              </div>
            </div>
          </div>

          {/* Recent History Table Enhanced - Claude Style */}
          <div className="lg:col-span-2 space-y-4 animate-fade-in-up animate-stagger-3">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-2xl font-bold text-claude-white flex items-center gap-3">
                <Clock className="w-6 h-6 text-claude-coral" />
                Activité Récente
              </h2>
              <Link 
                to="/history" 
                className="text-sm text-claude-coral hover:text-claude-coral-light transition-all flex items-center gap-2 group px-4 py-2 rounded-lg hover:bg-claude-coral/10 border border-transparent hover:border-claude-coral/30"
              >
                Voir l'historique 
                <ArrowRight className="w-4 h-4 transform group-hover:translate-x-1 transition-transform" />
              </Link>
            </div>

            <div className="bg-gradient-to-br from-claude-charcoal/90 via-claude-charcoal/80 to-claude-charcoal-dark/90 border border-claude-white/10 rounded-2xl overflow-hidden shadow-2xl relative backdrop-blur-sm">
              {/* Border gradient simplifié */}
              <div className="absolute inset-0 rounded-2xl bg-gradient-to-r from-purple-500/0 via-purple-500/5 to-blue-500/0 opacity-0 hover:opacity-100 transition-opacity duration-300 pointer-events-none" />
              
              {loading ? (
                <div className="p-12 text-center">
                  <div className="w-8 h-8 border-4 border-claude-coral/30 border-t-claude-coral rounded-full mx-auto mb-4 animate-spin" />
                  <p className="text-claude-grey-light font-mono">Chargement des données...</p>
                </div>
              ) : recentHistory.length === 0 ? (
                <div className="p-16 text-center border-dashed border-2 border-claude-white/10 m-6 rounded-xl bg-claude-charcoal-dark/30">
                  <AlertTriangle className="w-12 h-12 text-claude-grey mx-auto mb-4" />
                  <p className="text-claude-grey font-mono">Aucune donnée disponible</p>
                </div>
              ) : (
                <div className="divide-y divide-claude-white/10">
                  {recentHistory.map((entry, index) => (
                    <div 
                      key={entry.id}
                      className="animate-fade-in-up"
                      style={{ animationDelay: `${index * 0.1}s` }}
                    >
                      <HistoryEntryItem 
                        entry={entry}
                        getStatusColor={getStatusColor}
                      />
                    </div>
                  ))}
                </div>
              )}
            </div>
            
            {/* Action Feedback Messages */}
            {(actionError || actionSuccess) && (
              <div className={`mt-4 p-4 rounded-xl border ${
                actionSuccess 
                  ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400' 
                  : 'bg-red-500/10 border-red-500/30 text-red-400'
              } animate-fade-in`}>
                <div className="flex items-center gap-2">
                  {actionSuccess ? (
                    <CheckCircle className="w-5 h-5" />
                  ) : (
                    <XCircle className="w-5 h-5" />
                  )}
                  <span className="text-sm font-medium">
                    {actionSuccess || actionError}
                  </span>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

// --- Sous-composants améliorés ---

interface StatCardProps {
    title: string;
    value: number | string;
    icon: React.ComponentType<{ className?: string }>;
    color: string;
    bgGradient: string;
    borderColor: string;
    loading: boolean;
    trend?: 'up' | 'down' | 'neutral';
}

// Memoize les composants pour éviter les re-renders inutiles
const StatCard = memo(function StatCard({ 
    title, 
    value, 
    icon: Icon, 
    color, 
    bgGradient, 
    borderColor,
    loading,
    trend = 'neutral'
}: StatCardProps) {
    return (
        <div 
            className={`relative overflow-hidden bg-claude-charcoal/90 border ${borderColor} p-6 rounded-2xl shadow-xl group cursor-pointer hover-lift transition-smooth backdrop-blur-sm`}
        >
            {/* Simple gradient background */}
            <div 
                className={`absolute top-0 right-0 w-32 h-32 bg-gradient-to-br ${bgGradient} rounded-full opacity-5`}
            />
            
            <div className="flex justify-between items-start relative z-10">
                <div className="flex-1">
                    <p className="text-claude-grey-light text-xs font-medium uppercase tracking-wider mb-2">
                        {title}
                    </p>
                    <div className="flex items-baseline gap-2">
                        <h3 className="text-4xl font-bold text-claude-white font-mono">
                            {loading ? (
                                <span className="opacity-50">
                                    ...
                                </span>
                            ) : (
                                value
                            )}
                        </h3>
                        {trend !== 'neutral' && !loading && (
                            <div className={trend === 'up' ? 'text-emerald-400' : 'text-red-400'}>
                                <TrendingUp 
                                    className={`w-4 h-4 ${trend === 'down' ? 'rotate-180' : ''}`} 
                                />
                            </div>
                        )}
                    </div>
                </div>
                <div className={`p-4 rounded-xl bg-claude-charcoal-dark/50 ${color} shadow-lg transition-transform duration-300 group-hover:scale-110 group-hover:rotate-3`}>
                    <Icon className="w-7 h-7" />
                </div>
            </div>
            
            {/* Bottom accent line with animation */}
            <div
                className={`absolute bottom-0 left-0 right-0 h-1 bg-gradient-to-r ${bgGradient} opacity-0 group-hover:opacity-100 transition-opacity duration-300`}
            />
            {/* Shimmer effect on hover */}
            <div className="absolute inset-0 opacity-0 group-hover:opacity-100 animate-shimmer pointer-events-none" />
        </div>
    );
});

interface QuickActionCardProps {
    onClick: () => void;
    icon: React.ComponentType<{ className?: string }>;
    title: string;
    subtitle: string;
    description?: string;
    color: 'blue' | 'emerald' | 'purple';
    loading?: boolean;
    disabled?: boolean;
}

const QuickActionCard = memo(function QuickActionCard({ 
    onClick, 
    icon: Icon, 
    title, 
    subtitle, 
    description, 
    color,
    loading = false,
    disabled = false
}: QuickActionCardProps) {
    const colorConfig: Record<'blue' | 'emerald' | 'purple', {
        border: string;
        shadow: string;
        icon: string;
        bg: string;
        gradient: string;
    }> = {
        blue: {
            border: "group-hover:border-blue-500/50",
            shadow: "group-hover:shadow-blue-500/30",
            icon: "text-blue-400 bg-blue-500/10 group-hover:bg-blue-500/20",
            bg: "bg-blue-500/5",
            gradient: "from-blue-500/20 to-transparent"
        },
        emerald: {
            border: "group-hover:border-emerald-500/50",
            shadow: "group-hover:shadow-emerald-500/30",
            icon: "text-emerald-400 bg-emerald-500/10 group-hover:bg-emerald-500/20",
            bg: "bg-emerald-500/5",
            gradient: "from-emerald-500/20 to-transparent"
        },
        purple: {
            border: "group-hover:border-purple-500/50",
            shadow: "group-hover:shadow-purple-500/30",
            icon: "text-purple-400 bg-purple-500/10 group-hover:bg-purple-500/20",
            bg: "bg-purple-500/5",
            gradient: "from-purple-500/20 to-transparent"
        },
    };

    const config = colorConfig[color];

    return (
        <button
            onClick={onClick}
            disabled={disabled || loading}
            className={`relative flex items-center gap-4 p-5 bg-claude-charcoal/90 border border-claude-white/10 rounded-xl hover-lift transition-smooth shadow-lg group overflow-hidden backdrop-blur-sm ${config.border} ${config.shadow} ${
                disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'
            } ${loading ? 'animate-pulse' : ''}`}
        >
            {/* Background gradient */}
            <div
                className={`absolute inset-0 bg-gradient-to-r ${config.gradient} opacity-0 group-hover:opacity-100 transition-opacity duration-200`}
            />
            
            {/* Icon with animation */}
            <div className={`p-3.5 rounded-xl ${config.icon} transition-all duration-300 relative z-10 group-hover:scale-110 group-hover:-rotate-6`}>
                {loading ? (
                    <Activity className="w-6 h-6 animate-spin" />
                ) : (
                    <Icon className="w-6 h-6 transition-transform duration-300" />
                )}
            </div>
            
            <div className="flex-1 relative z-10 text-left">
                <h3 className="font-semibold text-claude-white group-hover:text-claude-white transition-colors mb-1">
                    {title}
                </h3>
                <p className="text-sm text-claude-grey-light group-hover:text-claude-grey font-medium">
                    {subtitle}
                </p>
                {description && (
                    <p className="text-xs text-claude-grey mt-1 group-hover:text-claude-grey-light">
                        {description}
                    </p>
                )}
            </div>
            
            <div className="relative z-10">
                {loading ? (
                    <Activity className="w-5 h-5 text-claude-coral animate-spin" />
                ) : (
                    <ArrowRight className="w-5 h-5 text-claude-grey group-hover:text-claude-coral opacity-0 group-hover:opacity-100 transform -translate-x-2 group-hover:translate-x-0 transition-all duration-300 group-hover:scale-110" />
                )}
            </div>
            
            {/* Bottom accent line */}
            <div
                className={`absolute bottom-0 left-0 right-0 h-0.5 bg-gradient-to-r ${config.gradient} opacity-0 group-hover:opacity-100 transition-opacity`}
            />
        </button>
    );
});

interface ActionCardProps {
    to: string;
    icon: React.ComponentType<{ className?: string }>;
    title: string;
    subtitle: string;
    description?: string;
    color: 'blue' | 'emerald' | 'purple';
}

const ActionCard = memo(function ActionCard({ to, icon: Icon, title, subtitle, description, color }: ActionCardProps) {
    const colorConfig: Record<'blue' | 'emerald' | 'purple', {
        border: string;
        shadow: string;
        icon: string;
        bg: string;
        gradient: string;
    }> = {
        blue: {
            border: "group-hover:border-blue-500/50",
            shadow: "group-hover:shadow-blue-500/30",
            icon: "text-blue-400 bg-blue-500/10 group-hover:bg-blue-500/20",
            bg: "bg-blue-500/5",
            gradient: "from-blue-500/20 to-transparent"
        },
        emerald: {
            border: "group-hover:border-emerald-500/50",
            shadow: "group-hover:shadow-emerald-500/30",
            icon: "text-emerald-400 bg-emerald-500/10 group-hover:bg-emerald-500/20",
            bg: "bg-emerald-500/5",
            gradient: "from-emerald-500/20 to-transparent"
        },
        purple: {
            border: "group-hover:border-purple-500/50",
            shadow: "group-hover:shadow-purple-500/30",
            icon: "text-purple-400 bg-purple-500/10 group-hover:bg-purple-500/20",
            bg: "bg-purple-500/5",
            gradient: "from-purple-500/20 to-transparent"
        },
    };

    const config = colorConfig[color];

    return (
        <Link to={to}>
            <div 
                className={`relative flex items-center gap-4 p-5 bg-claude-charcoal/90 border border-claude-white/10 rounded-xl hover-lift transition-smooth shadow-lg group overflow-hidden backdrop-blur-sm ${config.border} ${config.shadow}`}
            >
                {/* Background gradient */}
                <div
                    className={`absolute inset-0 bg-gradient-to-r ${config.gradient} opacity-0 group-hover:opacity-100 transition-opacity duration-200`}
                />
                
                {/* Icon with animation */}
                <div className={`p-3.5 rounded-xl ${config.icon} transition-all duration-300 relative z-10 group-hover:scale-110 group-hover:-rotate-6`}>
                    <Icon className="w-6 h-6 transition-transform duration-300" />
                </div>
                
                <div className="flex-1 relative z-10">
                    <h3 className="font-semibold text-claude-white group-hover:text-claude-white transition-colors mb-1">
                        {title}
                    </h3>
                    <p className="text-sm text-claude-grey-light group-hover:text-claude-grey font-medium">
                        {subtitle}
                    </p>
                    {description && (
                        <p className="text-xs text-claude-grey mt-1 group-hover:text-claude-grey-light">
                            {description}
                        </p>
                    )}
                </div>
                
                <div className="relative z-10">
                    <ArrowRight className="w-5 h-5 text-claude-grey group-hover:text-claude-coral opacity-0 group-hover:opacity-100 transform -translate-x-2 group-hover:translate-x-0 transition-all duration-300 group-hover:scale-110" />
                </div>
                
                {/* Bottom accent line */}
                <div
                    className={`absolute bottom-0 left-0 right-0 h-0.5 bg-gradient-to-r ${config.gradient} opacity-0 group-hover:opacity-100 transition-opacity`}
                />
            </div>
        </Link>
    );
});

interface SystemStatusCardProps {
    icon: React.ComponentType<{ className?: string }>;
    label: string;
    value: string;
    status: 'normal' | 'active' | 'secure' | 'warning';
}

const SystemStatusCard = memo(function SystemStatusCard({ icon: Icon, label, value, status }: SystemStatusCardProps) {
    const statusColors = {
        normal: 'text-blue-400 bg-blue-500/10 border-blue-500/20',
        active: 'text-emerald-400 bg-emerald-500/10 border-emerald-500/20',
        secure: 'text-purple-400 bg-purple-500/10 border-purple-500/20',
        warning: 'text-yellow-400 bg-yellow-500/10 border-yellow-500/20',
    };

    return (
        <div
            className={`p-3 rounded-lg bg-claude-charcoal/70 border ${statusColors[status]} hover-lift transition-smooth backdrop-blur-sm`}
        >
            <div className="flex items-center gap-2 mb-2">
                <Icon className="w-4 h-4" />
                <span className="text-xs text-claude-grey-light font-medium">{label}</span>
            </div>
            <p className="text-sm font-semibold font-mono text-claude-white">{value}</p>
        </div>
    );
});

interface ServiceStatusItemProps {
    label: string;
    ready: boolean;
    icon: React.ComponentType<{ className?: string }>;
}

const ServiceStatusItem = memo(function ServiceStatusItem({ label, ready, icon: Icon }: ServiceStatusItemProps) {
    return (
        <div className="flex items-center justify-between p-2 rounded-lg bg-claude-charcoal-dark/30 border border-claude-white/10 backdrop-blur-sm">
            <div className="flex items-center gap-2">
                <Icon className={`w-4 h-4 ${ready ? 'text-emerald-400' : 'text-red-400'}`} />
                <span className="text-sm text-claude-grey-light">{label}</span>
            </div>
            <div className="flex items-center gap-2">
                <div
                    className={`w-2 h-2 rounded-full ${ready ? 'bg-emerald-500 animate-pulse' : 'bg-red-500'}`}
                />
                <span className={`text-xs font-mono ${ready ? 'text-emerald-400' : 'text-red-400'}`}>
                    {ready ? 'ONLINE' : 'OFFLINE'}
                </span>
            </div>
        </div>
    );
});

// Composant optimisé pour les entrées d'historique
interface HistoryEntryItemProps {
  entry: HistoryEntry;
  getStatusColor: (status: HistoryEntry['status']) => string;
}

const HistoryEntryItem = memo(function HistoryEntryItem({ 
  entry, 
  getStatusColor
}: HistoryEntryItemProps) {
  const statusConfig = {
    completed: { icon: CheckCircle, glow: 'shadow-emerald-500/20', pulse: false },
    failed: { icon: XCircle, glow: 'shadow-red-500/20', pulse: false },
    running: { icon: Activity, glow: 'shadow-blue-500/30', pulse: true },
    pending: { icon: Clock, glow: 'shadow-yellow-500/20', pulse: false }
  };

  const config = statusConfig[entry.status] || statusConfig.pending;
  const StatusIcon = config.icon;

  return (
    <div className="group relative p-5 flex items-center justify-between transition-all duration-300 cursor-default overflow-hidden hover:bg-claude-charcoal-dark/30 animate-slide-in-right hover-lift">
      {/* Animated background gradient */}
      <div className="absolute inset-0 bg-gradient-to-r from-claude-coral/0 via-claude-coral/5 to-claude-white/0 opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
      
      {/* Scan line effect on hover */}
      <div className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none">
        <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-claude-coral/50 to-transparent animate-scan-line" />
      </div>

      {/* Left border accent - animated */}
      <div className={`absolute left-0 top-0 bottom-0 w-1 bg-gradient-to-b ${
        entry.status === 'completed' 
          ? 'from-emerald-400/0 via-emerald-400/50 to-emerald-400/0 group-hover:from-emerald-400/50 group-hover:via-emerald-400 group-hover:to-emerald-400/50'
          : entry.status === 'failed'
            ? 'from-red-400/0 via-red-400/50 to-red-400/0 group-hover:from-red-400/50 group-hover:via-red-400 group-hover:to-red-400/50'
            : entry.status === 'running'
              ? 'from-blue-400/0 via-blue-400/50 to-blue-400/0 group-hover:from-blue-400/50 group-hover:via-blue-400 group-hover:to-blue-400/50 animate-pulse'
              : 'from-yellow-400/0 via-yellow-400/50 to-yellow-400/0 group-hover:from-yellow-400/50 group-hover:via-yellow-400 group-hover:to-yellow-400/50'
      } transition-all duration-500`} />
      
      <div className="min-w-0 flex-1 mr-4 relative z-10">
        {/* Header with animated status indicator */}
        <div className="flex items-center gap-3 mb-3">
          <div className="relative">
            <span 
              className={`absolute inset-0 rounded-full ${
                entry.status === 'running' 
                  ? 'bg-blue-400 animate-ping opacity-75' 
                  : entry.status === 'completed' 
                    ? 'bg-emerald-400' 
                    : entry.status === 'failed'
                      ? 'bg-red-400'
                      : 'bg-yellow-400'
              } ${config.pulse ? 'animate-pulse' : ''}`}
            />
            <span 
              className={`relative w-2.5 h-2.5 rounded-full ${
                entry.status === 'running' 
                  ? 'bg-blue-400' 
                  : entry.status === 'completed' 
                    ? 'bg-emerald-400' 
                    : entry.status === 'failed'
                      ? 'bg-red-400'
                      : 'bg-yellow-400'
              } ${config.pulse ? 'animate-pulse' : ''} shadow-lg ${config.glow}`}
            />
          </div>
          <h4 className="font-semibold text-claude-grey-light truncate font-mono text-sm group-hover:text-claude-white transition-all duration-300 group-hover:tracking-wide">
            {entry.query}
          </h4>
        </div>

        {/* Metadata with icons */}
        <div className="flex items-center gap-4 text-xs text-claude-grey mb-3">
          <span className="flex items-center gap-1.5 group-hover:text-claude-grey-light transition-colors">
            <Clock className="w-3.5 h-3.5 group-hover:scale-110 transition-transform duration-300" />
            <span className="font-mono">{new Date(entry.timestamp).toLocaleTimeString()}</span>
          </span>
          <span className="px-2.5 py-1 rounded-md bg-claude-charcoal-dark/50 text-claude-grey-light border border-claude-white/10 font-mono text-xs group-hover:border-claude-coral/30 group-hover:bg-claude-coral/5 group-hover:text-claude-coral transition-all duration-300">
            {entry.target_agent || 'Router'}
          </span>
          {entry.generation_method && (
            <span className={`px-2.5 py-1 rounded-md font-mono text-xs border transition-all duration-300 group-hover:scale-105 ${
              entry.generation_method === 'RAG' 
                ? 'bg-blue-500/10 text-blue-400 border-blue-500/30 group-hover:bg-blue-500/20 group-hover:border-blue-500/50 group-hover:shadow-blue-500/20 group-hover:shadow-lg'
                : 'bg-claude-coral/10 text-claude-coral border-claude-coral/30 group-hover:bg-claude-coral/20 group-hover:border-claude-coral/50 group-hover:shadow-claude-coral/20 group-hover:shadow-lg'
            }`}>
              {entry.generation_method}
            </span>
          )}
        </div>

        {/* Command display with enhanced styling */}
        {entry.generated_command && (
          <div className="mt-3 p-3 bg-claude-charcoal-dark/40 rounded-lg border border-claude-white/10 group-hover:border-claude-coral/30 group-hover:bg-claude-charcoal-dark/60 transition-all duration-300 relative overflow-hidden backdrop-blur-sm">
            {/* Shimmer effect on hover */}
            <div className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500">
              <div className="absolute inset-0 bg-gradient-to-r from-transparent via-claude-coral/5 to-transparent -translate-x-full group-hover:translate-x-full transition-transform duration-1000" />
            </div>
            <p className="text-xs text-claude-grey mb-1.5 flex items-center gap-1.5 group-hover:text-claude-grey-light transition-colors">
              <Code className="w-3 h-3 group-hover:text-claude-coral transition-colors" />
              Commande générée:
            </p>
            <p className="text-xs font-mono text-claude-grey-light break-all relative z-10 group-hover:text-claude-white transition-colors">
              {entry.generated_command}
            </p>
          </div>
        )}
      </div>

      {/* Status badge with enhanced animations */}
      <div 
        className={`px-4 py-2 rounded-full text-xs font-semibold border relative z-10 transition-all duration-300 group-hover:scale-105 ${getStatusColor(entry.status)} ${
          entry.status === 'running' 
            ? 'animate-pulse shadow-lg shadow-blue-500/30' 
            : entry.status === 'completed'
              ? 'group-hover:shadow-lg group-hover:shadow-emerald-500/30'
              : entry.status === 'failed'
                ? 'group-hover:shadow-lg group-hover:shadow-red-500/30'
                : 'group-hover:shadow-lg group-hover:shadow-yellow-500/30'
        }`}
      >
        <div className="flex items-center gap-2">
          <StatusIcon className={`w-3.5 h-3.5 ${
            entry.status === 'running' ? 'animate-spin' : 'group-hover:scale-110 transition-transform duration-300'
          }`} />
          <span className="uppercase tracking-wider">{entry.status}</span>
        </div>
      </div>

      {/* Bottom accent line - animated */}
      <div className={`absolute bottom-0 left-0 right-0 h-0.5 bg-gradient-to-r ${
        entry.status === 'completed'
          ? 'from-emerald-500/0 via-emerald-500/50 to-emerald-500/0'
          : entry.status === 'failed'
            ? 'from-red-500/0 via-red-500/50 to-red-500/0'
            : entry.status === 'running'
              ? 'from-claude-coral/0 via-claude-coral/50 to-claude-coral/0'
              : 'from-claude-coral/0 via-claude-coral/30 to-claude-coral/0'
      } opacity-0 group-hover:opacity-100 transition-opacity duration-500`} />
    </div>
  );
});
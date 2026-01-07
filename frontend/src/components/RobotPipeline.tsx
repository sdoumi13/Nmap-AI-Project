import { memo, useMemo } from 'react';
import { 
  Brain, Network, Code, Cpu, Shield, 
  Activity,
  Zap, Database, Globe, Terminal, Layers, ArrowRight
} from 'lucide-react';
import { HealthStatus } from '../services/api';

interface RobotProps {
  id: string;
  name: string;
  description: string;
  icon: React.ComponentType<{ className?: string }>;
  status: 'idle' | 'active' | 'processing' | 'complete' | 'error';
  color: string;
  badge?: string;
  logs?: string[];
  technologies?: string[];
}

const Robot = memo(function Robot({ 
  name, 
  description, 
  icon: Icon, 
  status, 
  color,
  badge,
  logs = [],
  technologies = []
}: RobotProps) {
  // Calculate isAnimating directly from status to avoid synchronous state updates
  const isAnimating = status === 'active' || status === 'processing';

  const getStatusColors = (statusVal: string, colorVal: string) => {
    if (statusVal === 'idle') return 'from-claude-charcoal to-claude-charcoal-dark';
    if (statusVal === 'complete') return 'from-emerald-500 to-emerald-600';
    if (statusVal === 'error') return 'from-red-500 to-red-600';
    
    if (colorVal === 'coral') return 'from-claude-coral to-claude-coral/80';
    if (colorVal === 'purple') return 'from-purple-500 to-purple-500/80';
    if (colorVal === 'emerald') return 'from-emerald-500 to-emerald-500/80';
    return 'from-blue-500 to-blue-500/80';
  };

  const statusColors = {
    idle: getStatusColors('idle', color),
    active: getStatusColors('active', color),
    processing: getStatusColors('processing', color),
    complete: getStatusColors('complete', color),
    error: getStatusColors('error', color),
  };

  const statusGlow = {
    idle: '',
    active: 'shadow-lg shadow-coral/30',
    processing: 'shadow-lg shadow-coral/50 animate-glow-pulse',
    complete: 'shadow-lg shadow-emerald-500/50',
    error: 'shadow-lg shadow-red-500/50',
  };

  return (
    <div className="relative flex flex-col items-center group">
      {/* Robot Container */}
      <div className="relative">
        <div 
          className={`
            relative w-40 h-40 rounded-2xl 
            bg-gradient-to-br ${statusColors[status]}
            border-2 border-claude-white/20
            ${statusGlow[status]}
            transition-all duration-500
            ${isAnimating ? 'animate-robot-active' : 'animate-robot-idle'}
            hover:scale-110 hover:z-10
            transform-gpu
            mb-4
          `}
          style={{
            transformStyle: 'preserve-3d',
            boxShadow: status === 'processing' 
              ? `0 0 40px ${color === 'coral' ? 'rgba(255, 107, 107, 0.8)' : color === 'purple' ? 'rgba(139, 92, 246, 0.8)' : color === 'emerald' ? 'rgba(16, 185, 129, 0.8)' : 'rgba(59, 130, 246, 0.8)'}`
              : '0 10px 30px rgba(0, 0, 0, 0.4)',
          }}
        >
          {/* 3D Face Plate */}
          <div className="absolute inset-3 rounded-xl bg-claude-charcoal-dark/50 backdrop-blur-sm border border-claude-white/10">
            <div className="flex items-center justify-center h-full">
              <Icon className={`w-16 h-16 ${color === 'coral' ? 'text-claude-coral' : color === 'purple' ? 'text-purple-400' : color === 'emerald' ? 'text-emerald-400' : 'text-blue-400'} transition-transform duration-300 ${status === 'processing' ? 'animate-spin' : ''}`} />
            </div>
          </div>

          {/* Status Indicator */}
          <div className={`absolute -top-2 -right-2 w-6 h-6 rounded-full border-2 border-claude-charcoal ${
            status === 'active' || status === 'processing' 
              ? 'bg-claude-coral animate-pulse' 
              : status === 'complete'
              ? 'bg-emerald-500'
              : status === 'error'
              ? 'bg-red-500'
              : 'bg-claude-grey'
          }`} />

          {/* Badge */}
          {badge && (
            <div className="absolute -top-3 left-1/2 transform -translate-x-1/2 px-3 py-1 bg-claude-charcoal-dark/95 border border-claude-white/20 rounded-full shadow-lg">
              <span className="text-xs font-mono font-bold text-claude-white">{badge}</span>
            </div>
          )}

          {/* Processing Indicator */}
          {status === 'processing' && (
            <div className="absolute inset-0 rounded-2xl overflow-hidden">
              <div className={`absolute inset-0 bg-gradient-to-r from-transparent ${
                color === 'coral' ? 'via-claude-coral/30' : 
                color === 'purple' ? 'via-purple-500/30' : 
                color === 'emerald' ? 'via-emerald-500/30' : 
                'via-blue-500/30'
              } to-transparent animate-data-flow`} />
            </div>
          )}
        </div>

        {/* Robot Name */}
        <div className="text-center mt-2">
          <h3 className="text-sm font-bold text-claude-white mb-1">{name}</h3>
          <p className="text-xs text-claude-grey-light leading-tight max-w-[200px]">{description}</p>
        </div>

        {/* Technologies used */}
        {technologies.length > 0 && (
          <div className="flex gap-2 justify-center mt-3 flex-wrap">
            {technologies.map((tech, idx) => (
              <div key={idx} className="px-2 py-1 bg-claude-charcoal-dark/80 border border-claude-white/10 rounded text-xs font-mono text-claude-grey-light">
                {tech}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Robot Info Card on hover */}
      <div className="absolute top-full left-1/2 transform -translate-x-1/2 mt-4 w-64 opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none z-30">
        <div className="bg-claude-charcoal-dark/95 border border-claude-white/20 rounded-lg p-3 shadow-xl backdrop-blur-sm">
          <h4 className="text-sm font-bold text-claude-white mb-1">{name}</h4>
          <p className="text-xs text-claude-grey-light mb-2">{description}</p>
          {logs.length > 0 && (
            <div className="text-xs text-claude-grey font-mono max-h-20 overflow-y-auto">
              {logs.slice(-2).map((log, i) => (
                <div key={i} className="text-claude-coral/80">• {log}</div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
});

interface FlowArrowProps {
  active: boolean;
  color?: string;
  length?: string;
}

const FlowArrow = memo(function FlowArrow({ active, color = 'coral', length = 'w-32' }: FlowArrowProps) {
  const isCoral = color === 'coral';
  const lineColor = active 
    ? (isCoral ? 'from-claude-coral/60 via-claude-coral to-claude-coral/60' : 'from-purple-400/60 via-purple-400 to-purple-400/60')
    : 'from-claude-grey/20 via-claude-grey/30 to-claude-grey/20';
  const arrowIconColor = active 
    ? (isCoral ? 'text-claude-coral' : 'text-purple-400')
    : 'text-claude-grey/40';
  const packetColor = isCoral ? 'bg-claude-coral' : 'bg-purple-400';
  
  return (
    <div className={`relative ${length} h-1 flex items-center justify-center`}>
      {/* Flow line */}
      <div className={`absolute inset-0 bg-gradient-to-r ${lineColor} ${active ? 'animate-pulse' : ''}`} />
      
      {/* Animated data packet */}
      {active && (
        <div className={`absolute left-0 w-3 h-3 rounded-full ${packetColor} animate-pulse shadow-lg z-10`}>
          <div className={`absolute inset-0 rounded-full ${packetColor} animate-ping opacity-75`} />
        </div>
      )}
      
      {/* Arrow head */}
      <ArrowRight className={`absolute right-0 w-5 h-5 ${arrowIconColor} transition-colors duration-300`} />
    </div>
  );
});

interface TechnicalBadgeProps {
  label: string;
  value: string | number;
  icon: React.ComponentType<{ className?: string }>;
  status?: 'online' | 'offline' | 'warning';
}

const TechnicalBadge = memo(function TechnicalBadge({ label, value, icon: Icon, status = 'online' }: TechnicalBadgeProps) {
  const statusColor = {
    online: 'text-emerald-400 border-emerald-500/30 bg-emerald-500/10',
    offline: 'text-red-400 border-red-500/30 bg-red-500/10',
    warning: 'text-yellow-400 border-yellow-500/30 bg-yellow-500/10',
  };

  return (
    <div className={`flex items-center gap-2 px-3 py-2 rounded-lg border ${statusColor[status]} transition-all hover:scale-105`}>
      <Icon className="w-4 h-4" />
      <div className="flex flex-col">
        <span className="text-xs text-claude-grey font-mono">{label}</span>
        <span className="text-sm font-bold text-claude-white">{value}</span>
      </div>
    </div>
  );
});

interface RobotPipelineProps {
  currentStage?: number;
  query?: string;
  status?: Record<string, 'idle' | 'active' | 'processing' | 'complete' | 'error'>;
  logs?: Record<string, string[]>;
  healthStatus?: HealthStatus | null;
}

export default function RobotPipeline({ 
  currentStage = 0, 
  query = '',
  status = {},
  logs = {},
  healthStatus
}: RobotPipelineProps) {
  const selectedModel = useMemo<'mistral' | 'phi4' | 'lora-t5' | null>(() => {
    return status.router === 'active' ? 'mistral' : null;
  }, [status.router]);

  const robots = [
    {
      id: 'comprehension',
      name: 'Comprehension Agent',
      description: 'TF-IDF + SBERT Embeddings | Semantic Intent Analysis',
      icon: Brain,
      color: 'coral',
      badge: 'TF-IDF',
      defaultStatus: status.comprehension || 'idle',
      logs: logs.comprehension || [],
      technologies: ['NLP', 'Phi-4'],
    },
    {
      id: 'router',
      name: 'Complexity Router',
      description: 'Classification Engine | Easy/Medium/Hard Routing',
      icon: Network,
      color: 'purple',
      badge: 'ROUTER',
      defaultStatus: status.router || 'idle',
      logs: logs.router || [],
      technologies: ['Embedding'],
    },
    {
      id: 'mistral',
      name: 'Mistral LLM',
      description: 'Mistral Model | Complex Query Generation',
      icon: Cpu,
      color: 'purple',
      badge: 'LLM',
      defaultStatus: status.mistral || 'idle',
      logs: logs.mistral || [],
      technologies: [],
    },
    {
      id: 'phi4',
      name: 'Phi-4',
      description: 'Phi-4 Model | Efficient Generation',
      icon: Zap,
      color: 'purple',
      badge: 'PHI-4',
      defaultStatus: status.phi4 || 'idle',
      logs: logs.phi4 || [],
      technologies: [],
    },
    {
      id: 'lora-t5',
      name: 'LoRA T5',
      description: 'LoRA-tuned T5 | Fine-tuned Generation',
      icon: Code,
      color: 'purple',
      badge: 'LoRA',
      defaultStatus: status['lora-t5'] || 'idle',
      logs: logs['lora-t5'] || [],
      technologies: [],
    },
    {
      id: 'agent5',
      name: 'Agent 5 Supervisor',
      description: 'Self-Correction | Sandboxing | VM Execution',
      icon: Shield,
      color: 'emerald',
      badge: 'MCP',
      defaultStatus: status.agent5 || 'idle',
      logs: logs.agent5 || [],
      technologies: [],
    },
  ];

  return (
    <div className="relative w-full min-h-[700px] bg-gradient-to-br from-claude-charcoal-dark via-claude-charcoal to-claude-charcoal-dark rounded-2xl border border-claude-white/10 overflow-hidden flex flex-col">
      {/* Header Section - Reorganized to prevent overlap */}
      <div className="relative h-28 flex-shrink-0 border-b border-claude-white/10 px-6 py-4 flex flex-col gap-3 bg-claude-charcoal/50 backdrop-blur-sm z-30">
        {/* Top row: Title centered */}
        <div className="flex justify-center">
          <div className="flex items-center gap-3 px-6 py-2 bg-claude-charcoal-dark/90 backdrop-blur-sm rounded-xl border border-claude-white/20 shadow-xl">
            <Activity className="w-6 h-6 text-claude-coral animate-pulse" />
            <h2 className="text-xl font-bold text-claude-white">
              <span className="bg-clip-text text-transparent bg-gradient-to-r from-claude-coral via-claude-white to-claude-coral">
                5-Robot Agent Pipeline
              </span>
            </h2>
          </div>
        </div>

        {/* Bottom row: Technical Badges and Indicators */}
        <div className="flex items-center justify-between">
          {/* Left: Technical Badges */}
          <div className="flex flex-wrap gap-2">
            <TechnicalBadge 
              label="TF-IDF" 
              value={healthStatus?.local_agents?.comprehension_ready ? 'ACTIVE' : 'IDLE'}
              icon={Database}
              status={healthStatus?.local_agents?.comprehension_ready ? 'online' : 'offline'}
            />
            <TechnicalBadge 
              label="MCP" 
              value={healthStatus?.external_services?.agent5_mcp?.status === 'online' ? 'CONNECTED' : 'DISCONNECTED'}
              icon={Terminal}
              status={healthStatus?.external_services?.agent5_mcp?.status === 'online' ? 'online' : 'offline'}
            />
            <TechnicalBadge 
              label="LangChain" 
              value="KG"
              icon={Layers}
              status="online"
            />
            <TechnicalBadge 
              label="REST API" 
              value={healthStatus?.status === 'online' ? 'ONLINE' : 'OFFLINE'}
              icon={Globe}
              status={healthStatus?.status === 'online' ? 'online' : 'offline'}
            />
          </div>

          {/* Right: Knowledge Graph indicator */}
          <div className="text-xs text-claude-grey-light font-mono flex items-center gap-2">
            <Layers className="w-4 h-4 text-claude-coral" />
            <span>Knowledge Graph • REST API • Phi-4</span>
          </div>
        </div>
      </div>

      {/* Main Pipeline Area - Horizontal flow */}
      <div className="flex-1 relative overflow-x-auto overflow-y-hidden scrollbar-hide">
        {/* Background Grid Pattern */}
        <div 
          className="absolute inset-0 opacity-10"
          style={{
            backgroundImage: `
              linear-gradient(rgba(255, 255, 255, 0.1) 1px, transparent 1px),
              linear-gradient(90deg, rgba(255, 255, 255, 0.1) 1px, transparent 1px)
            `,
            backgroundSize: '40px 40px',
          }}
        />
        
        {/* Pipeline Container - Full horizontal layout */}
        <div className="relative min-w-[1600px] h-full py-12 px-12 flex items-center justify-between">
          {/* STEP 1: Comprehension Agent */}
          <div className="flex flex-col items-center relative z-10">
            <Robot
              id={robots[0].id}
              name={robots[0].name}
              description={robots[0].description}
              icon={robots[0].icon}
              status={currentStage >= 0 ? (currentStage === 0 ? 'processing' : 'complete') : 'idle'}
              color={robots[0].color}
              badge={robots[0].badge}
              logs={robots[0].logs}
              technologies={robots[0].technologies}
            />
          </div>

          {/* Arrow 1: Comprehension -> Router */}
          <FlowArrow active={currentStage >= 1} color="coral" length="w-24" />

          {/* STEP 2: Complexity Router */}
          <div className="flex flex-col items-center relative z-10">
            <Robot
              id={robots[1].id}
              name={robots[1].name}
              description={robots[1].description}
              icon={robots[1].icon}
              status={currentStage >= 1 ? (currentStage === 1 ? 'processing' : 'complete') : 'idle'}
              color={robots[1].color}
              badge={robots[1].badge}
              logs={robots[1].logs}
              technologies={robots[1].technologies}
            />
          </div>

          {/* Arrow 2: Router -> Models */}
          <FlowArrow active={currentStage >= 2} color="purple" length="w-24" />

          {/* STEP 3: Specialized Models (Vertical Stack with visual connections) */}
          <div className="flex flex-col items-center gap-8 relative z-10">
            {/* Connection line from Router to selected model */}
            {selectedModel && (
              <svg className="absolute left-[-6rem] top-0 bottom-0 w-24 h-full pointer-events-none" style={{ zIndex: 0 }}>
                {selectedModel === 'mistral' && (
                  <line
                    x1="100%"
                    y1="50%"
                    x2="0%"
                    y2="0%"
                    stroke="rgba(139, 92, 246, 0.5)"
                    strokeWidth="3"
                    strokeDasharray={currentStage >= 2 ? "0" : "5,5"}
                    className="transition-all duration-500"
                  />
                )}
                {selectedModel === 'phi4' && (
                  <line
                    x1="100%"
                    y1="50%"
                    x2="0%"
                    y2="50%"
                    stroke="rgba(139, 92, 246, 0.5)"
                    strokeWidth="3"
                    strokeDasharray={currentStage >= 2 ? "0" : "5,5"}
                    className="transition-all duration-500"
                  />
                )}
                {selectedModel === 'lora-t5' && (
                  <line
                    x1="100%"
                    y1="50%"
                    x2="0%"
                    y2="100%"
                    stroke="rgba(139, 92, 246, 0.5)"
                    strokeWidth="3"
                    strokeDasharray={currentStage >= 2 ? "0" : "5,5"}
                    className="transition-all duration-500"
                  />
                )}
              </svg>
            )}

            {/* Mistral LLM */}
            <div className={`relative ${selectedModel === 'mistral' ? '' : 'opacity-40'} transition-opacity duration-300`}>
              {selectedModel === 'mistral' && (
                <>
                  <div className="absolute -inset-3 rounded-full border-2 border-purple-400/50 animate-ping z-0" />
                  <div className="absolute -inset-2 rounded-full border border-purple-400/30 z-0" />
                </>
              )}
              <Robot
                id={robots[2].id}
                name={robots[2].name}
                description={robots[2].description}
                icon={robots[2].icon}
                status={selectedModel === 'mistral' && currentStage >= 2 ? (currentStage === 2 ? 'processing' : 'complete') : 'idle'}
                color={robots[2].color}
                badge={robots[2].badge}
                logs={robots[2].logs}
                technologies={robots[2].technologies}
              />
              {selectedModel === 'mistral' && currentStage >= 2 && (
                <div className="absolute -bottom-8 left-1/2 transform -translate-x-1/2 px-3 py-1 bg-purple-500/90 border border-purple-400 rounded text-xs font-mono text-white shadow-lg backdrop-blur-sm animate-pulse">
                  ACTIVE
                </div>
              )}
            </div>

            {/* Phi-4 */}
            <div className={`relative ${selectedModel === 'phi4' ? '' : 'opacity-40'} transition-opacity duration-300`}>
              {selectedModel === 'phi4' && (
                <>
                  <div className="absolute -inset-3 rounded-full border-2 border-purple-400/50 animate-ping z-0" />
                  <div className="absolute -inset-2 rounded-full border border-purple-400/30 z-0" />
                </>
              )}
              <Robot
                id={robots[3].id}
                name={robots[3].name}
                description={robots[3].description}
                icon={robots[3].icon}
                status={selectedModel === 'phi4' && currentStage >= 2 ? (currentStage === 2 ? 'processing' : 'complete') : 'idle'}
                color={robots[3].color}
                badge={robots[3].badge}
                logs={robots[3].logs}
                technologies={robots[3].technologies}
              />
              {selectedModel === 'phi4' && currentStage >= 2 && (
                <div className="absolute -bottom-8 left-1/2 transform -translate-x-1/2 px-3 py-1 bg-purple-500/90 border border-purple-400 rounded text-xs font-mono text-white shadow-lg backdrop-blur-sm animate-pulse">
                  ACTIVE
                </div>
              )}
            </div>

            {/* LoRA T5 */}
            <div className={`relative ${selectedModel === 'lora-t5' ? '' : 'opacity-40'} transition-opacity duration-300`}>
              {selectedModel === 'lora-t5' && (
                <>
                  <div className="absolute -inset-3 rounded-full border-2 border-purple-400/50 animate-ping z-0" />
                  <div className="absolute -inset-2 rounded-full border border-purple-400/30 z-0" />
                </>
              )}
              <Robot
                id={robots[4].id}
                name={robots[4].name}
                description={robots[4].description}
                icon={robots[4].icon}
                status={selectedModel === 'lora-t5' && currentStage >= 2 ? (currentStage === 2 ? 'processing' : 'complete') : 'idle'}
                color={robots[4].color}
                badge={robots[4].badge}
                logs={robots[4].logs}
                technologies={robots[4].technologies}
              />
              {selectedModel === 'lora-t5' && currentStage >= 2 && (
                <div className="absolute -bottom-8 left-1/2 transform -translate-x-1/2 px-3 py-1 bg-purple-500/90 border border-purple-400 rounded text-xs font-mono text-white shadow-lg backdrop-blur-sm animate-pulse">
                  ACTIVE
                </div>
              )}
            </div>

            {/* Connection line from selected model to Agent 5 */}
            {selectedModel && (
              <svg className="absolute right-[-6rem] top-0 bottom-0 w-24 h-full pointer-events-none" style={{ zIndex: 0 }}>
                {selectedModel === 'mistral' && (
                  <line
                    x1="0%"
                    y1="0%"
                    x2="100%"
                    y2="50%"
                    stroke="rgba(139, 92, 246, 0.5)"
                    strokeWidth="3"
                    strokeDasharray={currentStage >= 3 ? "0" : "5,5"}
                    className="transition-all duration-500"
                  />
                )}
                {selectedModel === 'phi4' && (
                  <line
                    x1="0%"
                    y1="50%"
                    x2="100%"
                    y2="50%"
                    stroke="rgba(139, 92, 246, 0.5)"
                    strokeWidth="3"
                    strokeDasharray={currentStage >= 3 ? "0" : "5,5"}
                    className="transition-all duration-500"
                  />
                )}
                {selectedModel === 'lora-t5' && (
                  <line
                    x1="0%"
                    y1="100%"
                    x2="100%"
                    y2="50%"
                    stroke="rgba(139, 92, 246, 0.5)"
                    strokeWidth="3"
                    strokeDasharray={currentStage >= 3 ? "0" : "5,5"}
                    className="transition-all duration-500"
                  />
                )}
              </svg>
            )}
          </div>

          {/* Arrow 3: Model -> Agent 5 */}
          <FlowArrow active={currentStage >= 3} color="purple" length="w-24" />

          {/* STEP 4: Agent 5 Supervisor */}
          <div className="flex flex-col items-center relative z-10">
            <Robot
              id={robots[5].id}
              name={robots[5].name}
              description={robots[5].description}
              icon={robots[5].icon}
              status={currentStage >= 3 ? (currentStage === 3 ? 'processing' : currentStage > 3 ? 'complete' : 'active') : 'idle'}
              color={robots[5].color}
              badge={robots[5].badge}
              logs={robots[5].logs}
              technologies={robots[5].technologies}
            />
          </div>
        </div>

        {/* Query Display - Bottom center */}
        {query && (
          <div className="absolute bottom-6 left-1/2 transform -translate-x-1/2 z-20">
            <div className="bg-claude-charcoal-dark/95 border border-claude-white/20 rounded-lg px-6 py-3 backdrop-blur-sm shadow-xl">
              <p className="text-xs text-claude-grey font-mono mb-1">Processing Query:</p>
              <p className="text-sm text-claude-white font-semibold font-mono">{query}</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
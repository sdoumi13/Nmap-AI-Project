import { ReactNode, useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { 
  LayoutDashboard, 
  Network, 
  History,
  Menu,
  X,
  Activity,
  CheckCircle2,
  Radio
} from 'lucide-react';

interface LayoutProps {
  children: ReactNode;
}

const navigation = [
  { name: 'Dashboard', href: '/', icon: LayoutDashboard },
  { name: 'Router', href: '/router', icon: Network },
  { name: 'Historique', href: '/history', icon: History },
];

export default function Layout({ children }: LayoutProps) {
  const location = useLocation();
  const [sidebarOpen, setSidebarOpen] = useState(false);

  return (
    <div className="min-h-screen bg-slate-950 text-white selection:bg-purple-500 selection:text-white overflow-hidden flex relative">
      {/* Static Background - No animations */}
      <div className="fixed inset-0 z-0 pointer-events-none">
        <div className="absolute top-0 right-0 w-96 h-96 bg-purple-900/10 rounded-full blur-2xl" />
        <div className="absolute bottom-0 left-0 w-[500px] h-[500px] bg-blue-900/8 rounded-full blur-2xl" />
        
        {/* Static grid pattern */}
        <div 
          className="absolute inset-0 opacity-[0.03]"
          style={{
            backgroundImage: `
              linear-gradient(rgba(139, 92, 246, 0.1) 1px, transparent 1px),
              linear-gradient(90deg, rgba(139, 92, 246, 0.1) 1px, transparent 1px)
            `,
            backgroundSize: '50px 50px'
          }}
        />
      </div>

      {/* Sidebar - No animations */}
      <aside className="hidden md:flex w-72 bg-slate-900/95 border-r border-slate-800/50 relative z-20 flex-col h-screen shadow-2xl">
        {/* Sidebar Header */}
        <div className="p-6 border-b border-slate-800/50">
          <div className="flex items-center space-x-3">
            <div className="relative">
              <div className="p-2.5 bg-gradient-to-tr from-purple-600 to-blue-600 rounded-xl shadow-lg relative overflow-hidden">
                {/* Logo SVG inline */}
                <svg className="h-6 w-6 text-white" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <defs>
                    <linearGradient id="sidebarLogoGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                      <stop offset="0%" style={{stopColor:"#ffffff",stopOpacity:1}} />
                      <stop offset="100%" style={{stopColor:"#e0e7ff",stopOpacity:1}} />
                    </linearGradient>
                  </defs>
                  {/* Network hexagon */}
                  <polygon points="32,12 42,18 42,28 32,34 22,28 22,18" 
                           fill="url(#sidebarLogoGradient)" 
                           stroke="#c4b5fd" 
                           strokeWidth="1.5"
                           opacity="0.95"/>
                  {/* Inner hexagon */}
                  <polygon points="32,18 38,22 38,28 32,32 26,28 26,22" 
                           fill="none" 
                           stroke="#ffffff" 
                           strokeWidth="1.5"
                           opacity="0.8"/>
                  {/* Center dot */}
                  <circle cx="32" cy="25" r="3" fill="#ffffff" opacity="0.9"/>
                  {/* Scan lines */}
                  <line x1="32" y1="32" x2="32" y2="4" stroke="#ffffff" strokeWidth="1.5" opacity="0.4"/>
                  <line x1="32" y1="32" x2="50" y2="20" stroke="#ffffff" strokeWidth="1.5" opacity="0.3"/>
                  <line x1="32" y1="32" x2="50" y2="44" stroke="#ffffff" strokeWidth="1.5" opacity="0.3"/>
                  <line x1="32" y1="32" x2="14" y2="20" stroke="#ffffff" strokeWidth="1.5" opacity="0.3"/>
                  <line x1="32" y1="32" x2="14" y2="44" stroke="#ffffff" strokeWidth="1.5" opacity="0.3"/>
                </svg>
              </div>
              {/* Glow effect */}
              <div className="absolute inset-0 bg-purple-400/20 rounded-xl blur-lg -z-10 opacity-0 group-hover:opacity-100 transition-opacity" />
            </div>
            <div>
              <h1 className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-white via-purple-200 to-blue-200">
                Nmap AI
              </h1>
              <div className="flex items-center gap-2 mt-1">
                <div className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                <p className="text-xs text-slate-400 font-mono">SOC v1.0</p>
              </div>
            </div>
          </div>
          
          {/* Status indicator */}
          <div className="mt-4 flex items-center gap-2 text-xs text-slate-400 font-mono">
            <Activity className="w-3 h-3 text-emerald-400" />
            <span>System Operational</span>
          </div>
        </div>

        <nav className="flex-1 p-4 space-y-2 overflow-y-auto">
          {navigation.map((item) => {
            const Icon = item.icon;
            const isActive = location.pathname === item.href;
            
            return (
              <Link
                key={item.name}
                to={item.href}
                className="block relative group"
                onClick={() => setSidebarOpen(false)}
              >
                {isActive && (
                  <div className="absolute inset-0 bg-gradient-to-r from-purple-600/20 via-purple-600/10 to-blue-600/10 rounded-xl border border-purple-500/30" />
                )}
                <div className={`
                  flex items-center space-x-3 px-4 py-3.5 rounded-xl transition-colors duration-150 relative z-10
                  ${isActive ? 'text-white' : 'text-slate-400 hover:text-white hover:bg-white/5'}
                `}>
                  <Icon className={`h-5 w-5 ${isActive ? 'text-purple-400' : 'text-slate-500 group-hover:text-purple-400'} transition-colors`} />
                  <span className="font-medium text-sm">{item.name}</span>
                  {isActive && (
                    <div className="ml-auto w-2 h-2 rounded-full bg-purple-400" />
                  )}
                </div>
              </Link>
            );
          })}
        </nav>

        {/* Footer */}
        <div className="p-4 border-t border-slate-800/50 space-y-3">
          <div className="bg-slate-800/60 rounded-xl p-4 border border-slate-700/50">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <div className="h-2 w-2 rounded-full bg-emerald-400" />
                <span className="text-xs text-slate-300 font-semibold">Security Status</span>
              </div>
              <CheckCircle2 className="w-4 h-4 text-emerald-400" />
            </div>
            <div className="space-y-2">
              <div className="flex items-center justify-between text-xs">
                <span className="text-slate-400">Threat Level</span>
                <span className="text-emerald-400 font-mono font-semibold">LOW</span>
              </div>
              <div className="flex items-center justify-between text-xs">
                <span className="text-slate-400">Active Scans</span>
                <span className="text-blue-400 font-mono">0</span>
              </div>
            </div>
          </div>
          
          {/* Terminal-style footer */}
          <div className="bg-slate-950/50 rounded-lg p-3 border border-slate-800/50 font-mono text-xs">
            <div className="flex items-center gap-2 text-slate-500">
              <Radio className="w-3 h-3 text-emerald-400" />
              <span>Connected to SOC</span>
            </div>
          </div>
        </div>
      </aside>

      {/* Mobile Sidebar */}
      {sidebarOpen && (
        <>
          <div
            onClick={() => setSidebarOpen(false)}
            className="fixed inset-0 bg-black/50 z-40 md:hidden"
          />
          <aside className="fixed left-0 top-0 bottom-0 w-72 bg-slate-900/95 border-r border-slate-800/50 z-50 flex flex-col md:hidden">
            <div className="p-6 border-b border-slate-800/50 flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <div className="p-2.5 bg-gradient-to-tr from-purple-600 to-blue-600 rounded-xl relative overflow-hidden">
                  {/* Logo SVG inline */}
                  <svg className="h-6 w-6 text-white" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <defs>
                      <linearGradient id="mobileLogoGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" style={{stopColor:"#ffffff",stopOpacity:1}} />
                        <stop offset="100%" style={{stopColor:"#e0e7ff",stopOpacity:1}} />
                      </linearGradient>
                    </defs>
                    {/* Network hexagon */}
                    <polygon points="32,12 42,18 42,28 32,34 22,28 22,18" 
                             fill="url(#mobileLogoGradient)" 
                             stroke="#c4b5fd" 
                             strokeWidth="1.5"
                             opacity="0.95"/>
                    {/* Inner hexagon */}
                    <polygon points="32,18 38,22 38,28 32,32 26,28 26,22" 
                             fill="none" 
                             stroke="#ffffff" 
                             strokeWidth="1.5"
                             opacity="0.8"/>
                    {/* Center dot */}
                    <circle cx="32" cy="25" r="3" fill="#ffffff" opacity="0.9"/>
                    {/* Scan lines */}
                    <line x1="32" y1="32" x2="32" y2="4" stroke="#ffffff" strokeWidth="1.5" opacity="0.4"/>
                    <line x1="32" y1="32" x2="50" y2="20" stroke="#ffffff" strokeWidth="1.5" opacity="0.3"/>
                    <line x1="32" y1="32" x2="50" y2="44" stroke="#ffffff" strokeWidth="1.5" opacity="0.3"/>
                    <line x1="32" y1="32" x2="14" y2="20" stroke="#ffffff" strokeWidth="1.5" opacity="0.3"/>
                    <line x1="32" y1="32" x2="14" y2="44" stroke="#ffffff" strokeWidth="1.5" opacity="0.3"/>
                  </svg>
                </div>
                <div>
                  <h1 className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-white to-slate-400">
                    Nmap AI
                  </h1>
                  <p className="text-xs text-slate-400 font-mono">SOC v1.0</p>
                </div>
              </div>
              <button
                onClick={() => setSidebarOpen(false)}
                className="p-2 text-slate-400 hover:text-white transition-colors"
                title="Close Menu"
                aria-label="Close Menu"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            <nav className="flex-1 p-4 space-y-2 overflow-y-auto">
              {navigation.map((item) => {
                const Icon = item.icon;
                const isActive = location.pathname === item.href;
                return (
                  <Link
                    key={item.name}
                    to={item.href}
                    onClick={() => setSidebarOpen(false)}
                    className="block relative group"
                  >
                    {isActive && (
                      <div className="absolute inset-0 bg-gradient-to-r from-purple-600/20 to-blue-600/10 rounded-xl border border-purple-500/30" />
                    )}
                    <div className={`
                      flex items-center space-x-3 px-4 py-3.5 rounded-xl transition-colors duration-150 relative z-10
                      ${isActive ? 'text-white' : 'text-slate-400 hover:text-white hover:bg-white/5'}
                    `}>
                      <Icon className={`h-5 w-5 ${isActive ? 'text-purple-400' : 'text-slate-500'}`} />
                      <span className="font-medium text-sm">{item.name}</span>
                    </div>
                  </Link>
                );
              })}
            </nav>
          </aside>
        </>
      )}

      {/* Main Content */}
      <main className="flex-1 relative z-10 overflow-y-auto h-screen bg-transparent">
        {/* Header */}
        <header className="h-16 flex items-center justify-between px-4 md:px-8 border-b border-slate-800/30 bg-slate-950/80 sticky top-0 z-30 shadow-lg">
          <div className="flex items-center gap-4">
            <button 
              onClick={() => setSidebarOpen(true)}
              className="md:hidden p-2 text-slate-400 hover:text-white transition-colors rounded-lg hover:bg-slate-800/50"
              title="Open Menu"
              aria-label="Open Menu"
            >
              <Menu className="h-5 w-5" />
            </button>
            <div className="flex items-center gap-2">
              <div className="hidden md:block w-1 h-4 bg-gradient-to-b from-purple-500 to-blue-500 rounded-full" />
              <div className="text-sm text-slate-400 font-mono flex items-center gap-2">
                <span className="text-slate-500">$</span>
                <span className="text-slate-300">
                  {location.pathname === '/' ? '/dashboard' : location.pathname}
                </span>
                <span className="text-purple-400">_</span>
              </div>
            </div>
          </div>
          
          <div className="flex items-center space-x-4">
            {/* Security indicator */}
            <div className="hidden md:flex items-center gap-2 px-3 py-1.5 rounded-lg bg-emerald-500/10 border border-emerald-500/20">
              <div className="w-2 h-2 rounded-full bg-emerald-400" />
              <span className="text-xs text-emerald-400 font-mono font-semibold">SECURE</span>
            </div>
            
            <button 
              className="p-2 text-slate-400 hover:text-white transition-colors rounded-lg hover:bg-slate-800/50 hidden md:block" 
              title="Menu Options"
              aria-label="Menu Options"
            >
              <Menu className="h-5 w-5" />
            </button>
          </div>
        </header>
        
        <div className="p-4 md:p-8 pb-20 max-w-7xl mx-auto relative">
          {children}
        </div>
      </main>
    </div>
  );
}

# Fichier: agent_1_router/run_router.py
import sys
import os

# Ajout du path pour les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_1_router.comprehension import ComprehensionAgent
from agent_1_router.complexity import ComplexityAgent

def main():
    print("="*70)
    print("🤖 NMAP-AI: INTELLIGENT GATEWAY")
    print("   Tapez 'exit' pour quitter.")
    print("="*70)

    try:
        # On instancie sans arguments, les classes gèrent les chemins
        comp_agent = ComprehensionAgent()
        complex_agent = ComplexityAgent()
        print("\n✅ SYSTÈME PRÊT.\n")
    except Exception as e:
        print(f"\n❌ Erreur critique au démarrage : {e}")
        return

    while True:
        try:
            user_query = input("\n📝 Intention (ex: 'scan port 80'): ").strip()
            
            if user_query.lower() in ['exit', 'q', 'quit']:
                print("👋 Au revoir !")
                break
            
            if not user_query: continue

            print(f"   🔍 Analyse...")

            # 1. Compréhension
            comp_result = comp_agent.analyze(user_query)
            
            if not comp_result['relevant']:
                print(f"   ❌ REJETÉ. Score: {comp_result['score']:.2f}")
                print(f"      Raison: {comp_result['reason']}")
                continue
            
            print(f"   ✅ PERTINENT. Score: {comp_result['score']:.2f}")
            if comp_result['best_match']:
                 print(f"      💡 Similaire à: {comp_result['best_match'].get('command', 'N/A')}")

            # 2. Complexité
            routing = complex_agent.classify(user_query)
            
            # Couleurs pour le terminal
            color = '\033[92m' if routing['level'] == 'Easy' else '\033[93m' if routing['level'] == 'Medium' else '\033[91m'
            reset = '\033[0m'
            
            print(f"   🔀 ROUTAGE: {color}{routing['target_agent']}{reset}")
            print(f"      Mots-clés: {routing.get('matched_keywords', [])}")

        except KeyboardInterrupt:
            print("\nArrêt.")
            break
        except Exception as e:
            print(f"Erreur: {e}")

if __name__ == "__main__":
    main()
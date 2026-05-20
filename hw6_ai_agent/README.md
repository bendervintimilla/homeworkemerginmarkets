# HW6 – AI Agent con CrewAI (opcional)

**Lecture 12** – arquitectura de Agentes de IA. Esta práctica está marcada como
**opcional**.

## Objetivo

Practicar el patrón "Crew" de [CrewAI](https://docs.crewai.com): definir varios
agentes con roles, objetivos, backstory y herramientas, encadenarlos en
tareas, y ejecutar el flujo.

## Estructura de la crew

| Agente | Rol | Tarea |
|---|---|---|
| Researcher | Senior RL Researcher | Investiga PPO y resume puntos clave |
| Analyst | ML Critical Analyst | Critica y señala pitfalls |
| Writer | Technical Writer | Sintetiza un brief de 1 página |

`Process.sequential` – cada tarea recibe el output de la anterior.

## Ejecución

```bash
pip install crewai crewai-tools

# Define la API key en un fichero .env:
echo 'OPENAI_API_KEY=sk-...' > .env

python -m hw6_ai_agent.research_crew
```

Sin clave, el script solo construye la crew para validar la estructura.

## Por qué importa

- En la nueva era post-DeepSeek-R1 / RLHF la disciplina del **Agent Engineering**
  consiste en alinear capacidades del LLM con expectativas del cliente.
- CrewAI captura el patrón mínimo: dividir un problema complejo en *roles*,
  cada uno con *contexto* y *herramientas*, secuenciados en *tareas*.
- Esta lab no entrena nada; ejercita la abstracción del agente.

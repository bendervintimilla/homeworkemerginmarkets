"""HW6 (opcional) – AI Agent con CrewAI.

Demostración mínima pero completa del patrón "crew" (varios agentes con roles
y herramientas que se coordinan):

    Researcher:  busca y resume información sobre algoritmos de RL.
    Analyst:     compara y crítica los hallazgos.
    Writer:      redacta un resumen ejecutivo final.

Esta práctica es opcional. La idea es entender la *estructura* de un agente:
rol, objetivo, backstory, herramientas, tareas, memoria.

Para ejecutar localmente necesitas una API key (OpenAI o cualquier LLM
soportado) en una variable `.env`. El módulo se ejecuta sin claves para
inspeccionar la estructura, pero `main()` requerirá la clave para correr.
"""

from __future__ import annotations

import os
from pathlib import Path


def build_crew(topic: str = "Proximal Policy Optimization (PPO) in deep RL"):
    """Define and return a CrewAI Crew. Lazily imports the package so the rest
    of the repo doesn't require crewai to be installed."""
    try:
        from crewai import Agent, Crew, Process, Task
    except ImportError as e:
        raise ImportError(
            "crewai is not installed. `pip install crewai crewai-tools` to run HW6."
        ) from e

    researcher = Agent(
        role="Senior RL Researcher",
        goal=(
            f"Investigar y resumir el estado del arte de '{topic}', identificando "
            "papers clave, intuiciones y limitaciones."
        ),
        backstory=(
            "Investigador con 10 años en deep RL, ha implementado PPO, SAC y DQN "
            "y publicado en NeurIPS. Es metódico y cita con precisión."
        ),
        allow_delegation=False,
        verbose=True,
    )

    analyst = Agent(
        role="ML Critical Analyst",
        goal=(
            "Cuestionar y validar los hallazgos del investigador, señalando trade-offs, "
            "supuestos ocultos y casos en los que el algoritmo falla."
        ),
        backstory=(
            "Ex-engineer in a frontier lab, especializado en debugging de "
            "training inestable y pitfalls de implementación de PPO."
        ),
        allow_delegation=False,
        verbose=True,
    )

    writer = Agent(
        role="Technical Writer",
        goal=(
            "Sintetizar la conversación en un resumen ejecutivo de 1 página en "
            "markdown, claro para alguien con conocimientos básicos de RL."
        ),
        backstory=(
            "Redactor técnico con habilidad para explicar matemáticas a un público "
            "general sin sacrificar el rigor."
        ),
        allow_delegation=False,
        verbose=True,
    )

    research_task = Task(
        description=(
            f"Resume los puntos clave de {topic}: motivación, ecuaciones principales "
            "(loss clipped, GAE), hiperparámetros canónicos y resultados típicos."
        ),
        expected_output="Lista de bullet points con citas a papers cuando aplique.",
        agent=researcher,
    )

    critique_task = Task(
        description=(
            "Lee el resumen del researcher y propón 3-5 críticas o pitfalls "
            "comunes (e.g. KL explosion, advantage normalization mal hecha, "
            "minibatching insuficiente)."
        ),
        expected_output="Lista de pitfalls con explicación breve y sugerencia de fix.",
        agent=analyst,
    )

    writeup_task = Task(
        description=(
            "Combina la investigación y la crítica en un brief ejecutivo de "
            "una página en markdown."
        ),
        expected_output="Documento markdown final.",
        agent=writer,
    )

    crew = Crew(
        agents=[researcher, analyst, writer],
        tasks=[research_task, critique_task, writeup_task],
        process=Process.sequential,
        verbose=True,
    )
    return crew


def main():
    if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("ANTHROPIC_API_KEY"):
        print("[HW6] AVISO: no hay OPENAI_API_KEY ni ANTHROPIC_API_KEY en el entorno.")
        print("       Define una clave en .env y vuelve a ejecutar.")
        print("       Solo construyo la crew para validar la estructura.")
        crew = build_crew()
        print(f"[HW6] Crew construida con {len(crew.agents)} agentes y "
              f"{len(crew.tasks)} tareas.")
        return

    crew = build_crew()
    print("[HW6] Lanzando crew...")
    result = crew.kickoff()
    out = Path("hw6_ai_agent/results/output.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(str(result))
    print(f"[HW6] Resultado guardado en {out}")


if __name__ == "__main__":
    main()

# Guía completa del routing y Cycling Coach

Estado documentado: rama `feature/enhance-training-data-integrations`, commit `2ca0edd`.

Esta guía describe toda la funcionalidad de routing ciclista construida desde la
Fase 1 hasta la primera integración de la Fase 12. Está pensada tanto como
referencia funcional como para preparar pruebas manuales desde ChatGPT, Claude o
un cliente MCP.

## 1. Qué resuelve actualmente

El sistema cubre cuatro niveles de uso:

1. Analizar una ruta ya definida y encontrar su mejor ventana de entrenamiento.
2. Generar varias rutas circulares desde un origen, analizarlas y recomendar la
   mejor sesión completa.
3. Traducir perfiles de entrenamiento de alto nivel a parámetros auditables.
4. Combinar tiempo disponible, perfil, ruta, clima y exportación GPX mediante el
   Cycling Coach.

Las cuatro herramientas públicas son:

| Herramienta | Uso principal |
| --- | --- |
| `icu_find_best_cycling_training_window` | Analizar una ruta definida por puntos ordenados. |
| `icu_find_cycling_training_route` | Generar, filtrar, analizar y rankear rutas circulares. |
| `icu_find_profiled_cycling_training_route` | Buscar rutas usando perfiles `steady_climb` o `sweet_spot_climb`. |
| `icu_find_cycling_coach_route` | Planificar una sesión según el tiempo disponible y ejecutar el routing perfilado. |

## 2. Arquitectura general

```text
Petición MCP
   ↓
Validación de parámetros
   ↓
Geocoding y snapping del origen o puntos
   ↓
Generación de rutas cycling-road en OpenRouteService
   ↓
Filtros duros de viabilidad
   ↓
Deduplicación geométrica
   ↓
Timeline, elevación, extras y maniobras
   ↓
Análisis de warmup, bloque(s), cooldown y ruta completa
   ↓
Ranking lexicográfico determinista
   ↓
Clima y luz opcionales para la mejor ruta
   ↓
Serialización MCP y GPX opcional
```

No existe un score ponderado opaco. Los hard constraints deciden si una ruta es
elegible y el ranking ordena únicamente las candidatas elegibles.

## 3. Evolución por fases

### Fase 1 — Mejor ventana en una ruta conocida

Responde a: «Ya sé por dónde quiero ir; encuentra el mejor tramo para entrenar».

Construye una ruta `cycling-road` pasando por una lista ordenada de lugares o
coordenadas. Después crea un timeline continuo y busca ventanas de 20, 30 y 40
minutos por defecto, dentro del rango de inicio solicitado.

Analiza, entre otros elementos:

- distancia y duración efectiva;
- desnivel positivo y negativo;
- ritmo de ascenso y descenso normalizado por hora;
- gradiente neto;
- superficie y porcentaje de asfalto;
- porcentaje de carretera o vía ciclista;
- porcentaje de footway;
- suitability ciclista de ORS;
- maniobras e interrupciones.

Puede aplicar requisitos opcionales al bloque. Todos están desactivados por
defecto, por lo que no se inventan thresholds.

### Fase 2 — Generación y selección de rutas

Responde a: «Salgo desde aquí, quiero aproximadamente esta distancia y necesito
un bloque entrenable; genera alternativas y elige la mejor».

Genera entre 2 y 10 rutas circulares con semillas deterministas de ORS. La
distancia es una preferencia de generación, no una garantía. Por defecto se
rechazan rutas cuya desviación absoluta supere el 50 %.

La respuesta incluye:

- `best_route`;
- alternativas elegibles ordenadas;
- número de candidatas generadas y supervivientes de cada filtro;
- geometría GeoJSON;
- análisis del bloque de entrenamiento;
- metadatos y parámetros aplicados.

### Fase 3 — Calidad de la sesión completa

La ruta deja de evaluarse solo por su mejor subida. Se divide semánticamente en:

```text
SALIDA → WARMUP → TRAINING → COOLDOWN → LLEGADA
```

Para warmup y cooldown se calculan:

- duración y distancia;
- desnivel positivo y negativo;
- desnivel por hora;
- gradiente neto;
- maniobras por hora;
- roundabouts e interrupciones;
- asfalto, carretera/vía ciclista, footway y suitability;
- esfuerzo vertical aproximado.

La calidad global de ruta resume superficies, tipos de vía, pendientes,
maniobras, rotondas, giros bruscos e interrupciones. Estos análisis reutilizan el
mismo timeline, geometría, elevación y extras del bloque central.

También existen hard constraints opcionales e independientes para warmup y
cooldown:

- máximo desnivel positivo por hora;
- máximo gradiente neto;
- máximo de maniobras por hora;
- mínimo porcentaje de asfalto;
- máximo porcentaje de footway.

Si un threshold queda en `null`, no se aplica.

### Fase 4 — Deduplicación geométrica

Evita devolver varias rutas esencialmente iguales aunque ORS las haya generado
con semillas diferentes.

Comportamiento por defecto:

- remuestreo de geometría cada 100 m;
- puntos a 50 m o menos se consideran coincidentes;
- umbral de solapamiento simétrico del 90 %;
- se conserva la primera candidata según el orden determinista.

El solapamiento es simétrico para no considerar duplicada una ruta corta que solo
esté contenida parcialmente en otra más larga. Puede desactivarse usando
`deduplication_overlap_threshold_percentage=null`.

### Fase 5 — Objetivo de duración total

`target_duration_minutes` permite expresar una duración total deseada además de
la distancia. La duración sigue siendo una estimación del perfil ORS, no una
predicción personalizada del tiempo real del atleta.

La respuesta muestra desviación absoluta y porcentual. La duración actúa como
tie-breaker explícito después de la calidad de entrenamiento, warmup, cooldown y
ruta completa, y antes del ajuste de distancia.

`max_duration_deviation_percentage` es un hard constraint opcional y no tiene
valor arbitrario por defecto.

Actualmente no existe generación basada solo en duración: ORS necesita una
longitud para crear el round trip. Hacerlo sin distancia requeriría velocidad
esperada o un modelo histórico del atleta.

### Fase 6 — Exportación GPX

Con `include_gpx=true`, `best_route` incluye un GPX 1.1 codificado en base64. Las
alternativas no incorporan GPX para evitar respuestas innecesariamente grandes.

El GPX contiene:

- un track segment con latitud, longitud y elevación cuando está disponible;
- `TRAINING START` y `TRAINING END` en sesiones continuas;
- `BLOCK 1 START`, `BLOCK 1 END`, etc. en sesiones repetidas.

No realiza otra llamada a ORS: reutiliza la geometría ya obtenida. Para usarlo,
se decodifica `content_base64` y se guarda el resultado como archivo `.gpx`.

### Fase 7 — Entrenamientos multi-bloque

Permite repetir un bloque de igual duración mediante `training_repetitions`, con
recuperaciones mínima y máxima opcionales.

Ejemplo: 3 × 12 minutos con recuperaciones de 4–6 minutos.

Cada bloque debe cumplir individualmente los requisitos duros del training
window. La búsqueda cronológica está acotada y es determinista. Se serializan:

- todos los bloques y recuperaciones;
- trabajo total y distancia de trabajo;
- desnivel y balance de ascenso;
- maniobras e interrupciones;
- variación entre los ritmos de ascenso de los bloques;
- calidad de las recuperaciones.

Los tiempos se ajustan a puntos reales de la geometría, por lo que la suma
efectiva puede diferir ligeramente de la duración nominal.

### Fase 8 — Historial de carreteras nuevas/recorridas

No forma parte de la solución final actual. Se descartó conscientemente porque
aportaba poco valor para el caso de uso y añadía una capa importante de
persistencia, matching geométrico e integración histórica.

No se usa el historial del atleta para premiar carreteras nuevas ni penalizar
carreteras ya recorridas.

### Fase 9 — Restricciones geográficas seguras

Se exponen las restricciones nativas de ORS para ciclismo:

- `ferries`;
- `fords`;
- `steps`.

Se validan antes de llamar a ORS y quedan reflejadas en metadata. No se admiten
polígonos GeoJSON arbitrarios de exclusión porque requerirían validaciones de
topología, extensión, área y relación con la distancia de la ruta.

### Fase 10 — Clima y luz diurna

Al proporcionar `departure_time` con fecha ISO 8601 y offset UTC explícito, se
consulta Open-Meteo para el origen y el intervalo estimado de la ruta ganadora.

`data.external_context` incluye:

- hora local de salida y llegada estimada;
- horas de forecast muestreadas;
- temperatura y sensación térmica mínima/máxima;
- probabilidad y cantidad de precipitación;
- viento, rachas y direcciones;
- códigos meteorológicos WMO;
- amanecer y puesta de sol;
- indicación de si toda la sesión ocurre con luz diurna;
- avisos factuales de lluvia, tormenta o falta de luz.

El clima es informativo: no modifica eligibility ni ranking. No existen límites
inventados de temperatura o viento. Si Open-Meteo falla, la ruta se devuelve
igualmente con `available=false` y `forecast_unavailable`.

No se ofrece meteorología por segmento, viento frontal relativo, tráfico en vivo,
cierres de carretera ni garantía de seguridad.

### Fase 11 — Perfiles de entrenamiento

#### `steady_climb`

Busca un bloque continuo de subida estable. Si se indica
`work_duration_minutes`, usa esa duración. Si se omite, compara 20, 30 y 40
minutos. El inicio se busca por defecto entre los minutos 20 y 30.

#### `sweet_spot_climb`

Busca tres bloques sostenidos de subida. Por defecto configura:

- 3 × 12 minutos;
- recuperación de 5–8 minutos;
- horizonte de inicio de bloques entre los minutos 20 y 180.

El horizonte amplio permite representar el segundo y tercer bloque. El nombre
del perfil expresa la estructura prevista, pero no verifica potencia ejecutada ni
aplica automáticamente zonas fisiológicas.

Ambos perfiles devuelven `data.training_profile` con objetivo, parámetros
resueltos, intención de ranking, constraints aplicados y razones. No duplican el
motor de routing.

### Fase 12 — Primera integración del Cycling Coach

`icu_find_cycling_coach_route` añade una capa de planificación explícita:

- recibe perfil, origen, distancia y tiempo total disponible;
- valida todo antes de llamadas externas;
- si no hay `target_duration_minutes`, utiliza todo el tiempo disponible como
  objetivo de ruta;
- si existe un target explícito, exige que no supere el tiempo disponible;
- delega al routing perfilado existente;
- activa GPX por defecto;
- añade `data.coach_plan` con las decisiones y su justificación.

No añade buffers ocultos ni adapta automáticamente la sesión según readiness,
fatiga o wellness. Esa adaptación requiere una política explícita que todavía no
se ha definido.

## 4. Cómo funciona el ranking

El ranking es lexicográfico, auditable y determinista. A alto nivel sigue este
orden:

1. calidad del training window o secuencia de bloques;
2. limpieza y facilidad del warmup;
3. facilidad del cooldown;
4. calidad global de la ruta;
5. ajuste al objetivo de duración, si existe;
6. ajuste a la distancia;
7. semilla determinista.

En multi-bloque se prioriza primero el bloque más débil, después consistencia,
balance de ascenso, interrupciones, recuperaciones e inicio. De esta manera un
bloque excelente no oculta otros bloques deficientes.

## 5. Hard constraints frente a preferencias

Un hard constraint elimina candidatas antes del ranking. Ejemplos:

- desviación máxima de distancia;
- desviación máxima de duración;
- porcentaje mínimo de asfalto;
- máximo footway;
- máximo descenso o maniobras en el bloque;
- máximos de gradiente, ascenso o maniobras en warmup/cooldown;
- features nativas de ORS que deben evitarse.

Las métricas que no tienen un threshold explícito solo participan en el ranking o
se muestran como información. Salvo la tolerancia de distancia documentada, los
thresholds de calidad opcionales están desactivados por defecto.

## 6. Ejemplos de uso

### Ruta conocida: mejor bloque continuo

Petición natural:

> Analiza una ruta desde Son Moix, pasando por Esporles y Valldemossa, y encuentra
> el mejor bloque continuo de 30 minutos que empiece después de 20 minutos.

Argumentos orientativos:

```json
{
  "locations": ["Son Moix, Palma", "Esporles", "Valldemossa"],
  "start_time_min_minutes": 20,
  "start_time_max_minutes": 60,
  "durations_minutes": [30]
}
```

### Generar una ruta circular sencilla

```json
{
  "start_location": "39.589985,2.630108",
  "target_distance_km": 50,
  "training_durations_minutes": [20, 30, 40],
  "candidate_count": 4
}
```

### Exigir un calentamiento limpio

```json
{
  "start_location": "Son Moix, Palma",
  "target_distance_km": 45,
  "training_durations_minutes": [30],
  "max_warmup_maneuvers_per_hour": 8,
  "max_warmup_elevation_gain_rate_m_per_hour": 350,
  "max_warmup_footway_percentage": 0
}
```

Estos valores son decisiones del usuario, no defaults del sistema. Si ninguna
ruta los cumple, la respuesta será `not_found`.

### Tres bloques con recuperación

```json
{
  "start_location": "39.589985,2.630108",
  "target_distance_km": 60,
  "training_durations_minutes": [12],
  "training_repetitions": 3,
  "recovery_min_minutes": 5,
  "recovery_max_minutes": 8,
  "training_start_time_min_minutes": 20,
  "training_start_time_max_minutes": 180
}
```

### Sweet spot con clima y GPX

```json
{
  "profile": "sweet_spot_climb",
  "start_location": "Son Moix, Palma",
  "target_distance_km": 60,
  "departure_time": "2026-08-14T08:00:00+02:00",
  "include_gpx": true,
  "avoid_features": ["ferries", "fords", "steps"]
}
```

### Cycling Coach con 2 h 15 min disponibles

```json
{
  "profile": "sweet_spot_climb",
  "start_location": "Son Moix, Palma",
  "target_distance_km": 60,
  "available_time_minutes": 135,
  "departure_time": "2026-08-14T08:00:00+02:00",
  "candidate_count": 4,
  "include_gpx": true,
  "avoid_features": ["ferries", "fords", "steps"]
}
```

En este caso `coach_plan.target_duration_minutes` será 135 y
`duration_source` será `available_time`.

### Cycling Coach reservando margen explícito

```json
{
  "profile": "steady_climb",
  "start_location": "Son Moix, Palma",
  "target_distance_km": 50,
  "available_time_minutes": 150,
  "target_duration_minutes": 120,
  "work_duration_minutes": 35,
  "include_gpx": false
}
```

El sistema no inventa el margen: el usuario declara 150 minutos disponibles y
un objetivo de ruta de 120.

## 7. Estructura orientativa de la respuesta

Según el tool y los parámetros, `data` puede contener:

```json
{
  "best_route": {
    "route": {},
    "best_training_window": {},
    "training_block_sequence": {},
    "warmup": {},
    "cooldown": {},
    "route_quality": {},
    "gpx": {
      "format": "GPX 1.1",
      "encoding": "base64",
      "size_bytes": 0,
      "content_base64": "..."
    }
  },
  "alternatives": [],
  "training_profile": {},
  "coach_plan": {},
  "external_context": {}
}
```

La forma exacta depende de si la sesión es continua o repetida y de si se han
solicitado clima y GPX. `metadata` documenta filtros, requisitos y conteos de
candidatas.

## 8. Casos de error relevantes

- `configuration_error`: falta la API key de OpenRouteService.
- `validation_error`: parámetros incompatibles, negativos o thresholds inválidos.
- `location_resolution_error`: el lugar no se puede geocodificar o hacer snap.
- `not_found`: ninguna ruta o ventana cumple todos los hard constraints.
- `api_error`: error de ORS u otro proveedor externo.
- `internal_error`: error inesperado protegido por el envelope MCP.

Una ausencia de forecast no invalida la ruta; aparece como contexto externo no
disponible.

## 9. Plan recomendado de pruebas manuales

1. Probar `steady_climb` con coordenadas exactas y sin clima ni GPX.
2. Repetir la misma petición y confirmar resultado determinista.
3. Activar GPX, decodificarlo e importarlo en una aplicación de navegación.
4. Añadir una salida futura y revisar forecast, amanecer y puesta de sol.
5. Probar `sweet_spot_climb` y comprobar tres bloques y dos recuperaciones.
6. Usar un hard constraint deliberadamente estricto y confirmar `not_found`.
7. Probar un lugar por nombre y después las mismas coordenadas exactas.
8. Comparar `candidate_count=2` y `candidate_count=5`.
9. Probar un objetivo de duración y revisar su desviación.
10. Ejecutar el Cycling Coach con tiempo disponible y comprobar `coach_plan`.

Para aislar problemas, empezar con coordenadas evita ambigüedad de geocoding.
Después conviene probar nombres reales para validar la experiencia conversacional.

## 10. Límites y expectativas correctas

- La distancia circular de ORS es aproximada.
- La duración es una estimación ORS, no una predicción individual.
- `sweet_spot` describe estructura, no valida potencia ni zona fisiológica.
- No se modelan tráfico en vivo ni cierres.
- No se usan carreteras históricas o nuevas.
- No hay rutas basadas exclusivamente en tiempo sin una distancia.
- No hay adaptación automática por fatiga/readiness.
- No hay polígonos arbitrarios de exclusión.
- El clima no altera el ranking.
- La recomendación no constituye una garantía de seguridad vial.

## 11. Estado de validación

En `2ca0edd`:

- tests específicos del Cycling Coach, perfiles y transporte: 16 passed;
- suite completa: 560 passed y 5 fallos preexistentes;
- los cinco fallos conocidos corresponden a HR/wellness y no a routing;
- el MCP de producción carga la rama de feature y registró 65 herramientas en
  modo `delete_mode=none` después del despliegue.

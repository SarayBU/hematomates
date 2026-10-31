# Hematomates · Proba ti a simular o sangue

Páxina web interactiva da charla «Hematomates» (Matemáticas, habelas hainas 2026, Facultade de Matemáticas, USC).
Simula o fluxo do sangue nun vaso elástico en 1D e compara a solución numérica coa solución exacta dun problema
de Riemann.

- **Modelo:** conservación da masa e balance do momento cunha lei do tubo elástica, sen rozamento.
- **Método:** volumes finitos explícito con condicións de contorno transparentes. Fluxos: Godunov exacto,
  Lax–Friedrichs e FORCE.
- **Código:** baseado no código MATLAB de S. Busto.

## Contido

| Ruta | Que é |
|---|---|
| `index.html`, `js/app.js` | A páxina para o móbil |
| `js/blood1d.js` | O modelo e o esquema numérico en JavaScript |
| `python/blood1d.py` | A versión de referencia en Python |
| `python/check.py` | Comprobacións: converxencia á solución exacta e estabilidade segundo a CFL |
| `tests/make_ref.py`, `tests/check.html` | Comparan a versión JavaScript coa de Python |

## Probala no ordenador

```bash
python -m http.server 8000
```

e abrir `http://localhost:8000/` (as comprobacións, en `http://localhost:8000/tests/check.html`).

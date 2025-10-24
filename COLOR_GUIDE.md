# Color System & Guide — Professional UI Color Schemes

This document explains a practical, accessible, and designer/developer-friendly approach to creating and implementing professional color schemes for UI/UX.

## Quick summary
- Purpose: colors communicate brand, hierarchy, affordance, and feedback.
- Keep palettes small: 1–2 primary hues, 1 accent, and a neutral ramp for text, surfaces and borders.
- Use HSL-based tokens to generate consistent tonal ramps.
- Always test accessibility (WCAG contrast) and color-blind simulations.

## Contract — inputs / outputs / success
- Inputs: brand hue (hex or HSL), platform targets (web/mobile), accessibility level (AA/AAA).
- Outputs: neutral ramp, primary tonal ramp, semantic colors (success/warn/error), CSS tokens, light/dark variants, and a QA checklist.
- Success: components readable and usable (contrast, focus, non-color indicators) across devices and vision conditions.

## Core principles
- Purpose-first: assign roles (primary action, accent, backgrounds, text).
- Hierarchy: use contrast and saturation to indicate importance.
- Consistency: define tokens like `primary-600`, `neutral-900`, and reuse them.
- Emotional fit: saturation and hue change perceived tone (energetic vs. conservative).
- Build for light and dark modes from the start.

## Step-by-step workflow
1. Gather brand hue(s) and audience info. Aim for one strong anchor hue.
2. Build a neutral ramp (9 stops) from near-black to near-white for text/background separation.
3. Generate primary tonal ramp by adjusting lightness in HSL while keeping hue & saturation stable.
4. Add a single accent if needed for highlights.
5. Define semantic colors (success/warning/error/info) with appropriate contrast.
6. Create hover/active/disabled variants (systematic offsets in lightness/saturation).
7. Implement tokens in CSS/JSON and wire to components.
8. Test contrast, color blindness, and interactive states.

## Accessibility rules (practical)
- Contrast ratios: normal text >= 4.5:1 (AA), large text >= 3:1, enhanced >= 7:1 (AAA).
- Don't rely on color alone — add icons, text, or shapes for state.
- Test color blindness (protanopia, deuteranopia, tritanopia) with simulators.
- Tools: WebAIM Contrast Checker, Axe, Lighthouse, Stark (Figma), Color Oracle.

## Implementation patterns (developer friendly)
- Use design tokens / CSS variables and keep naming semantic (e.g., `--bg`, `--text-primary`, `--primary`).
- Prefer HSL or HSL channel values so you can change lightness easily for ramps.
- Systematic states: hover = darker by ~6–8% L, pressed = darker again; disabled = desaturate + raise L.
- Keep visible focus styles (3–4px ring) with sufficient contrast.

## Example tokens & CSS (already added to `static/styles.css`)
- A `:root` HSL token set with neutral ramp, primary variables, semantic colors, and a `@media (prefers-color-scheme: dark)` override.
- Component helpers included: `.btn`, `.btn-primary`, `.card-surface`, `.chip`, `.status-success`.

## How to choose palettes
- Start from the anchor (brand) hue. Use HSL lightness shifts to make tints and shades.
- For neutrals, use near-zero saturation grays to avoid color casts.
- Avoid highly saturated backgrounds with small text.

## Testing & automation
- Automated checks: Axe, Lighthouse, Storybook accessibility add-ons.
- Visual regression: Percy or Chromatic.
- Programmatic color tools: chroma.js, tinycolor, polished for generating ramps.

## Quick QA checklist
- [ ] Body text contrast >= 4.5:1
- [ ] Large text contrast >= 3:1
- [ ] All interactive elements have focus states visible
- [ ] Color not the only means to convey status
- [ ] UI usable under major color blindness types
- [ ] Dark mode tested and tuned

## Recommended resources
- Material Design color system — token examples
- WebAIM Contrast Checker — ratio testing
- Coolors / Adobe Color — palette generators
- Stark plugin (Figma/Sketch) — accessibility testing
- Color Oracle / Coblis — simulators

## Next steps I can help with
- Generate a tuned palette from your brand hex and include ready tokens.
- Produce a Tailwind config or JSON design-token file.
- Scan a page or screenshot for contrast issues and propose fixes.

---

If you want a palette generated now, paste a brand hex (e.g., `#2563EB`) and I'll return a light/dark token set plus small component examples and contrast checks.

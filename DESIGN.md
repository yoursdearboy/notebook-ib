---
version: alpha
name: Bioinformatics Institute
description: Reconstructed design draft for the public Institute website.
colors:
  text-body: '#3c3c50'
  heading-section: '#2d528a'
  text-photo-hero: '#ffffff'
  link-en: '#3a81e8'
  background-en-content: '#f5f5f5'
  divider-accordion: '#eeeeee'
typography:
  body:
    fontFamily: Open Sans
    fontWeight: 400
  accordion-label:
    fontFamily: Open Sans
    lineHeight: 1.35
rounded:
  accordion-row: 0px
spacing:
  content-max: 1200px
  content-inset: 20px
---

## Overview

This reconstructed draft describes the Bioinformatics Institute’s public site, which introduces the Institute and explains its education programs.

## Colors

Use deep blue for section headings on the program pages and dark slate for standard body copy. On the English landing page, distinguish inline links with a brighter blue and set the About and Education sections on a continuous light-gray background.

Set white copy over a dark overlay on photographic heroes, as on the English landing and bioinformatics program pages. The biostatistics hero uses dark text instead.

## Typography

Scale hero titles and accordion labels down at narrow widths and let long labels wrap. Keep the accordion label weight specific to the page: light for the bioinformatics program FAQ and semibold for biostatistics disclosures.

## Layout

Keep section backgrounds full bleed and center the main content grid within them. At narrow widths, let standard text and disclosure rows fill the available column up to the side insets.

## Components

- **Program navigation:** Show section links as a horizontal row on desktop and move them behind a menu button on narrow screens.
- **Program accordions:** Place the disclosure control at the trailing edge of a flat, divider-separated row. Use a plus icon in the collapsed state and open the answer directly below its trigger.

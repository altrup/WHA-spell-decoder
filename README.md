# WHA-spell-decoder
A VLM spell decoder for WHA, going to hand write the relevant training code (everything else probably not)

Plan is to have the model input the spell, then output a description of the spell, then the predicted effect

i.e.

```md
# Description
Fire glyph in center, 4 column glyphs facing towards center
# Justification
Water sigil sets the element. Bolt signs project it as arrows. Directional signs set the heading.
# Effect
Shoots fire upwards
```

```md
# Description
A single sign of pulling, arrow pointing toward the center. No sigil, no seal circle.
# Justification
Without a sigil there is no element to act on. Alone, this sign only modifies a spell.
# Effect
Not a complete spell. As a modifier it pulls the target toward the seal.
```

## Planned Model
https://huggingface.co/google/gemma-4-E2B
https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-gemma-4

## Data
going to base it off rules at https://witchhatatelier.telepedia.net/wiki/Signs_Explained
Some data at https://witch-hat-atelier.fandom.com/wiki/Spells, but prob gonna need to make more
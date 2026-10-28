# WHA-spell-decoder
A VLM spell decoder for WHA, going to hand write the relevant training code (everything else probably not)

Plan is to have the model input the spell (size normalized to frame using the circle; maybe), then output a description of the spell, then the predicted effect

i.e.

```md
# Description
Fire glyph in center, 4 column glyphs facing towards center
# Effect
Shoots fire upwards
```

## Planned Model
https://huggingface.co/google/gemma-4-E2B
https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-gemma-4

## Data
going to base it off rules at https://witchhatatelier.telepedia.net/wiki/Signs_Explained
Some data at https://witch-hat-atelier.fandom.com/wiki/Spells, but prob gonna need to make more
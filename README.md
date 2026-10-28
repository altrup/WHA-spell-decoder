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

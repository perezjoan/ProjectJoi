# Wan first/last-frame prompts for the Joi clips

Every clip is pinned at both ends: give the model the exact png as the first frame and the exact png as the last
frame, never let it choose an end. 512×512, 16 fps. Transition in: 49 frames (3 s). Return: 25 frames (1.5 s).
Loop: 49 frames, first = last = the still.

## Common positive (prepend to every prompt)

```
Photorealistic film still brought to life, a young woman from the shoulders up, dark brown hair with a straight
fringe, gold black and red brocade high-collar jacket, dim blurred room, soft light from the left. Locked camera,
no camera movement, no cut, the head stays centered in frame. She is silent: her lips do not move as if speaking,
no talking, no lip sync, no mouth flapping. One slow continuous change of expression only, smooth and natural,
consistent face and identity from first frame to last.
```

## Common negative (use on every clip)

```
talking, speaking, lip sync, lips moving as if speaking, mouth opening and closing repeatedly, singing, chewing,
mumbling, subtitles, captions, text, watermark, logo, camera movement, zoom, pan, cut, scene change, second person,
extra person, hands in frame, extra fingers, deformed hands, deformed face, face changing identity, different
woman, blurry, low quality, jpeg artifacts, overexposed, oversaturated, grey washed out, flicker, jitter, morphing
background, changing clothes, changing hair, glitch, distorted, ugly, disfigured
```

For the **idle loops** remove nothing from the list above but add to the positive: "almost still, breathing only".
Do not add "static" or "still image" to the negative for loops, since a loop is meant to be nearly still.

---

## n002 angry

**edge__n001__n002__v1** (first n001.png, last n002.png, 49 frames)
```
Over three seconds her calm face hardens into anger: the brows pull down and together, the stare fixes and
hardens, the jaw sets, the lips press thin and stay closed, the chin lifts slightly, the shoulders square. Slow,
continuous, no speech.
```
**edge__n002__n001__v1** (first n002.png, last n001.png, 25 frames)
```
Her anger fades: the brows level out, the jaw loosens, the pressed lips soften, the stare eases into a calm steady
gaze, the shoulders relax. Quick and smooth, no speech.
```
**loop__n002__v1** (first = last = n002.png, 49 frames)
```
Almost still, holding the angry expression: slow breathing, one blink near the middle of the clip, the hard stare stays
fixed; at rest and identical to the first frame at the start and the end. No speech.
```

## n003 puzzled

**edge__n001__n003__v1**
```
Over three seconds her calm face turns puzzled: one eyebrow lifts, the head tilts a little to one side, the lips
purse, the eyes narrow and search just past the camera. Slow, continuous, no speech.
```
**edge__n003__n001__v1**
```
The raised eyebrow settles, the head straightens, the pursed lips relax, the eyes return to a calm steady gaze.
Quick and smooth, no speech.
```
**loop__n003__v1**
```
Almost still, holding the puzzled look: slow breathing, one blink near the middle, the eyes drift slightly as if thinking,
the eyebrow stays raised; at rest and identical to the first frame at the start and the end. No speech.
```

## n004 happy

**edge__n001__n004__v1**
```
Over three seconds a warm smile grows on her calm face: the mouth opens into a broad smile showing her teeth, the
eyes brighten and crinkle, the cheeks lift, the head rises a touch. Slow, continuous, no speech.
```
**edge__n004__n001__v1**
```
The smile softens and closes, the eyes settle, the cheeks lower, the face returns to a calm steady gaze. Quick and
smooth, no speech.
```
**loop__n004__v1**
```
Almost still, holding the warm smile: slow breathing, one blink near the middle, the smile holds; at rest and identical to
the first frame at the start and the end. No speech.
```

## n005 crying

**edge__n001__n005__v1**
```
Over three seconds her calm face crumples into tears: the brows draw up at the inner ends, the eyes fill and
glisten, a tear spills down the cheek, the lips tremble and part slightly, the chin puckers, the head lowers a
little. Slow, continuous, no speech.
```
**edge__n005__n001__v1**
```
The tears stop, the brows level, the trembling lips close, the head lifts, the face settles back to a calm steady
gaze. Quick and smooth, no speech.
```
**loop__n005__v1**
```
Almost still, holding the tearful expression: slow uneven breathing, the eyes glisten, one slow blink near the middle, the
lips tremble faintly; at rest and identical to the first frame at the start and the end. No speech.
```

## n006 shocked

**edge__n001__n006__v1**
```
Over three seconds shock takes her calm face: the eyes fly wide, the brows shoot up, the lips part in a small gasp,
the chin draws back, the shoulders rise. Slow at first then sudden, continuous, no speech.
```
**edge__n006__n001__v1**
```
The wide eyes return to normal, the brows lower, the parted lips close, the shoulders drop, the face settles back to
calm. Quick and smooth, no speech.
```
**loop__n006__v1**
```
Almost still, holding the shocked expression: shallow breathing, the wide eyes stay wide with one quick blink near the
middle, the parted lips do not move; at rest and identical to the first frame at the start and the end. No speech.
```

## n007 worried

**edge__n001__n007__v1**
```
Over three seconds worry creeps into her calm face: the brows knit and lift at the inner ends, the gaze drops and
drifts to one side, the lips press together, the shoulders draw in. Slow, continuous, no speech.
```
**edge__n007__n001__v1**
```
The brows level out, the gaze lifts and steadies on the camera, the pressed lips relax, the shoulders open, the face
returns to calm. Quick and smooth, no speech.
```
**loop__n007__v1**
```
Almost still, holding the worried look: slow breathing, one blink near the middle, the lowered gaze wavers slightly; at rest
and identical to the first frame at the start and the end. No speech.
```

## n008 laughing

**edge__n001__n008__v1**
```
Over three seconds her calm face breaks into laughter: the smile spreads, the head tips back a little, the eyes
squeeze shut, the mouth opens wide in a silent laugh showing teeth, the shoulders shake. Continuous, no speech, no
sound, laughing not talking.
```
**edge__n008__n001__v1**
```
The laugh settles, the eyes open, the head comes level, the open mouth closes into a calm face, the shoulders still.
Quick and smooth, no speech.
```
**loop__n008__v1**
```
Almost still, holding the laugh: eyes stay shut, mouth stays open in a silent laugh, the shoulders shake gently, the head
stays tipped back; no blink; at rest and identical to the first frame at the start and the end. No speech, no talking.
```

## n009 wink

**edge__n001__n009__v1**
```
Over three seconds a playful wink: her left eye closes in a wink, a small knowing smile spreads with the lips
closed, the head tilts slightly, the other eye stays bright on the camera. Slow, continuous, no speech.
```
**edge__n009__n001__v1**
```
The winking eye opens, the knowing smile fades, the head straightens, the face returns to a calm steady gaze. Quick
and smooth, no speech.
```
**loop__n009__v1**
```
Almost still, holding the wink: the closed eye stays closed, the open eye stays open with no blink, the small closed-lip
smile holds, slow breathing, the faintest head movement; at rest and identical to the first frame at the start and the end.
No speech.
```

## n001 neutral

**loop__n001__v1** (first = last = n001.png, 49 frames)
```
Almost still, at rest: slow breathing, one blink near the middle of the clip, the faintest head movement; the face is at rest
and identical to the first frame at the start and the end. Lips relaxed, not moving. No speech.
```

---

## Checking a clip before it goes in

- First and last frames match the stills exactly (no drift in crop, hair or jacket). Both ends are pinned, so the seam
  risk in a loop is a blink or a movement still in progress near the last frame, forced to snap back: keep motion in the
  middle and rest at both ends.
- She never mouths words. If a clip still talks, add "lips stay closed" to the positive for that clip and
  "open mouth" to the negative, unless the expression itself needs an open mouth (laughing, shocked, crying).
- 49 frames for transitions and loops, 25 for returns, 16 fps; the durations in `clip_descriptions.json` assume
  3.06 s and 1.56 s.

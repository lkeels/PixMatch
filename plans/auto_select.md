# Auto-select

Status: **building**. Step 1 (measuring quality) is done. The Auto-select menu item and button exist but are disabled.

## Goal

One press goes through every shown group and plans what to do with it, the way you would by hand: keep the best
copy, mark the rest, and where the best copy is in the wrong folder, move it to the right one. Nothing happens to any
file until you press **Apply**, and you can change the plan before then.

## Two kinds of reason

- **Quality** decides *which file* to keep: resolution, compression (file size), sharpness. Objective: one copy can be
  *clearly better*, or several can be *about equal*.
- **Preference** decides between copies of about equal quality, and *where* the kept file belongs: colour over black
  and white, higher in the folder list, preferred format. A preference never overrules a clearly better copy on its
  own.

Folder order: a group's images are ranked by the listed folder they're under (an image in a subfolder counts as its
listed parent). Higher in the folder list is higher priority for keeping, but it isn't the final authority.

## How a group is decided

1. **Never touched:** IGNORED images, and images inside zips (they can't be deleted or moved).
2. **Find the best-quality copy.** One copy may be clearly better than every other, or several may be about equal.
3. **One clearly better copy:**
   - It's black and white (or sepia) and a colour copy exists: **conflict, the group is left alone**, for you to decide.
     (A sharper black and white copy against a colour one is your call.)
   - It's in the group's highest-ranked folder: keep it, mark the rest.
   - A copy is in a higher-ranked folder: plan a **MOVE TO** that folder, replacing the copy there, and mark the rest.
     Example: Tom Daley is first in the list, Unsorted below it. A much bigger copy of a Tom Daley picture is in
     Unsorted: it moves into the Tom Daley folder and replaces the small copy there.
4. **Several about equal:** preferences decide, in order:
   1. Colour beats black and white or sepia (at similar quality, black and white is sacrificial: it can always be made
      from the colour copy).
   2. Higher in the folder list.
   3. Preferred format (PNG or JPG over WebP).
   4. Bigger file.
   5. Still tied: exact copies in the same folder keep one (see Decisions, 2); otherwise the group is left alone.
5. Groups with an animated image are left alone (as in VisDeDupe: too ambiguous to decide automatically).

### Names of moved files

A moved file takes the name of the copy it replaces, since that's the name in the folder it belongs in, unless that
name looks random and the moved file's name looks like it was written by someone:

- Random-looking: camera and generated names (`IMG_8842`, `DSC02794`, `DSC_0054_2`), long hex or letter-and-digit
  strings (`587edaef10ead2cc...`, `GA1vsExWsAAmo1W`, `FfI0fvxWYAAnKz9`), site prefixes with ids (`tumblr_...`).
- Written: words, spaces, names (`Tom Daley beach`, `Lilac Whimsy`).

When it isn't sure, it keeps the replaced copy's name. The extension is the moved file's own (`.png` stays `.png`).
Moving never replaces any other file (as now: " (2)" is added if the name is taken).

## Using it

1. **Auto-select** warns that it will clear every current mark (MARKED, and any plan), and suggests acting on them
   first. IGNORED images stay ignored. Pressing it again later gives the same warning.
2. It runs with progress and Cancel (like regrouping), on every group, not just the page shown.
3. Tiles get banners:
   - **MARKED**: will be deleted.
   - **MOVE TO**: will be moved into a higher folder, replacing a copy there. Hovering shows the folder, and the file it
     replaces.
4. A summary dialog explains: what will happen ("Delete 40 files; move 3 better copies into their folders, replacing
   the copies there"), how many groups were left alone and why (conflicts, too close to call, animated), what each
   banner means and how to see where a MOVE TO goes, and that nothing has happened yet and any tile can be changed
   before applying.
5. An **Apply** button appears. It carries out the whole plan after a final confirmation listing the deletes and the
   replacements (the Recycle Bin may be off, so the list matters).

The right-click menu also gets "Replace the copy in *folder* with this", for doing the same by hand.

## Decisions

Settled 2026-10-20.

1. **Clearly better:** at least 25% more pixels; or at the same resolution, a clearly bigger file. A starting point, to
   be tuned on a real library (build step 5).
2. **Exact copies in the same folder** (same picture, same quality, same folder): keep one, the one with the more
   readable name, then the first; mark the rest.
3. **MOVE TO by hand:** only Auto-select and the right-click "Replace the copy in *folder* with this" set it. Clicking a
   tile still cycles KEEP, MARKED, IGNORED (clicking a MOVE TO tile goes back to KEEP).
4. **Delete and Move while a plan is waiting:** disabled until the plan is applied or cleared, so there's one way to
   carry it out. A **Clear plan** button sits next to Apply.

## Build steps

1. Measure quality during the scan (resolution and file size are free; sharpness and colour from the image the scan
   already reads), so Auto-select doesn't reopen 260k files. **Done** (b4e8c78): `ImageQuality`, from
   `ImageMatcher.quality(path)`. Hashes unchanged; costs ~1ms an image. In 800 library images: 80 grey
   (colourfulness < 2), 13 tinted (colour variety < 1.5), 707 colour, 24 between tinted and colour (variety 1.5-3) to
   look at when tuning.
2. The decision rules, tested on made-up groups covering every case above.
3. MOVE TO: the state, its banner and hover, Apply carrying out deletes and replacements, and the right-click
   "Replace...".
4. Auto-select itself: warning, progress and Cancel, the summary dialog.
5. Tuning "clearly better" and the name check on your library.

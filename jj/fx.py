"""fx: hand-drawn effects kit (effects animator).  `from jj import fx`.

Conventions.  f = frame (24 fps), t = seconds unless it says 0..1.  Functions taking `cel` draw vector shapes into an ink.Cel you own and return it; functions returning FX give
(premultiplied RGBA layer, additive light) -> `base = fx.put(base, FX)`.  Everything is deterministic in (f, seed).  Most drawings change on twos by themselves.

core   put merge hot glow_poly thin band Path teeth leaf tongue ink_broken noise noise_img displace
fire   flame_stream flames flame_ring fire_core embers fire_arrow fire_column   (palettes PAL, BLUEPAL)
curse  cursed aura limb_poly                                               (palettes CURSE, BLACKBLUE, VIOLET, BLOOD)
flash  black_flash black_flash_layers bolt_tree bolts_draw kite_star shards radial_wedges distort_ring ragged_ring impact_tone impact_frame higanbana slash_marks slash_fx cut_plate
break  rock debris ground_proj ground_crack puffs billow smoke_burst dust_skirt smoke_trail shockwave shatter glass_cells
orb    orb_blue orb_red orb_purple purple_merge purple_trail pinch void black_sun glyph
air    rain splash splashes snow petals ash motes mist
ink    ink_burst ink_runs ink_curtain ink_pool ink_wash paper_burn
"""
from .fx_core import *
from .fx_fire import *
from .fx_curse import *
from .fx_flash import *
from .fx_break import *
from .fx_orb import *
from .fx_air import *
from .fx_ink import *

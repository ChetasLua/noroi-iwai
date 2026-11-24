// Scene rooms: synthetic impulse responses (early reflections placed by hand from the set's geometry + band-wise diffuse tail).
//   getRoom(name) -> {l, r};  send level and distance behaviour live in SCENES (mix.mjs).
import { room } from './sfx_dsp.mjs';
export const ROOMS = {
  void:      { t60: 4.2, bands: { 63: 1.3, 125: 1.25, 250: 1.1, 1000: .75, 2000: .5, 4000: .3, 8000: .15 }, pre: .045, build: .12, hicut: 3600, early: [[61, .1, -.5, 1800], [97, .08, .6, 1500]] },           // a1: black, enormous, nothing near
  ink:       { t60: .5, bands: { 125: 1.2, 2000: .7, 4000: .5 }, pre: .006, hicut: 4200, early: [[9, .3, -.6, 3500], [13, .26, .7, 3000], [22, .18, .2, 2500]] },                                             // a2, f1 close: charcoal chamber
  paper:     { t60: .38, bands: { 125: .8, 4000: .8 }, pre: .003, hicut: 6000, early: [[4.1, .3, .4], [7.3, .22, -.5], [11.9, .16, .1]] },                                                                   // a3, g4: a desk in a small wooden room
  hill:      { t60: .5, tail: .25, bands: { 63: .5, 4000: .5 }, pre: .02, hicut: 4000, early: [[2.2, .14, 0, 3000], [340, .05, -.6, 1200], [470, .03, .7, 900]] },                                             // b1: grass under you, far valley answer
  slope:     { t60: .55, bands: { 4000: .6 }, pre: .008, hicut: 4600, early: [[4, .2, 0, 4000], [19, .16, -.8, 3000], [27, .14, .8, 2800], [88, .05, .4, 1500]] },                                              // b2: road, walls both sides
  comic:     { t60: .42, bands: { 125: .6 }, pre: .002, hicut: 5200, early: [[3.1, .26, -.5], [5.7, .22, .5], [8.8, .15, 0]] },                                                                                // b3, b4: close dry comedy stage
  street:    { t60: .9, bands: { 125: 1.2, 4000: .5, 8000: .3 }, pre: .009, hicut: 4600, early: [[11, .24, -.8, 4000], [17, .22, .8, 3600], [31, .14, -.4, 2800], [46, .1, .5, 2400]] },                       // b5: side street, shop fronts
  high:      { t60: .7, tail: .25, bands: { 4000: .4 }, pre: .03, hicut: 3000, early: [[3, .12, 0, 3000]] },                                                                                                   // b6: railing above the city
  shibuya:   { t60: 3.0, bands: { 63: 1.2, 125: 1.2, 2000: .7, 4000: .4, 8000: .2 }, pre: .05, build: .06, hicut: 3400, early: [[58, .2, -.8, 2400], [83, .18, .8, 2200], [141, .12, -.3, 1600], [196, .1, .5, 1400]] }, // c1
  subway:    { t60: 2.5, bands: { 63: .8, 125: 1.1, 250: 1.15, 4000: .5, 8000: .25 }, pre: .012, build: .03, hicut: 4200, early: [[14, .26, -.7, 3600], [21, .24, .7, 3400], [39, .18, 0, 2800], [66, .16, -.5, 2400], [93, .13, .5, 2000]] }, // c2
  close:     { t60: .28, bands: { 125: .7 }, pre: .002, hicut: 4200, early: [[2.6, .18, -.4, 3600], [4.4, .14, .5, 3200]] },                                                                                   // c3, c4 inserts
  ruin:      { t60: 1.4, bands: { 63: 1.1, 125: 1.2, 2000: .75, 4000: .5, 8000: .3 }, pre: .022, build: .02, hicut: 4200, early: [[23, .22, -.8, 3000], [34, .2, .8, 2800], [57, .14, -.2, 2200], [91, .1, .4, 1800]] }, // d1-d4, h: broken street
  sky:       { t60: 1.8, tail: .6, bands: { 4000: .4, 8000: .2 }, pre: .09, build: .2, hicut: 2600, early: [[420, .1, -.5, 1200], [610, .07, .6, 1000]] },                                                      // d5: far above the city
  infinite:  { t60: 7.0, bands: { 63: .8, 4000: .6, 8000: .3 }, pre: .06, build: .5, hicut: 4200 },                                                                                                             // d6
  tiled:     { t60: 2.4, bands: { 63: .6, 125: .8, 500: 1.1, 1000: 1.15, 2000: 1.0, 4000: .65, 8000: .35 }, pre: .012, build: .015, hicut: 5200, early: [[13, .3, -.5, 5000], [19, .28, .6, 5000], [33, .2, 0, 4200], [52, .16, -.6, 3600]] }, // e1
  embers:    { t60: 1.0, bands: { 2000: .6, 4000: .35 }, pre: .015, hicut: 3000, early: [[18, .16, -.6, 2400], [29, .13, .6, 2000]] },                                                                         // e2 first half
  beach:     { t60: .3, tail: .1, pre: .02, hicut: 4000 },                                                                                                                                                     // e2 second half
  water:     { t60: 1.7, bands: { 63: 1.2, 500: .8, 1000: .5, 2000: .25, 4000: .1 }, pre: .004, hicut: 900 },                                                                                                   // e3
  lamp:      { t60: 1.5, bands: { 125: 1.1, 4000: .45 }, pre: .02, hicut: 3600, early: [[27, .18, -.8, 2600], [41, .15, .8, 2400]] },                                                                          // e4
  crater:    { t60: 1.1, tail: .5, bands: { 4000: .4 }, pre: .03, hicut: 3000, early: [[64, .1, -.6, 1800], [92, .08, .6, 1600]] },                                                                            // e5
  temple:    { t60: 4.6, bands: { 63: 1.1, 125: 1.15, 2000: .7, 4000: .4, 8000: .2 }, pre: .035, build: .06, hicut: 3600, early: [[41, .16, -.6, 2600], [58, .14, .6, 2400], [97, .1, 0, 2000]] },               // f1 f2: water under a vast roof
  inkcity:   { t60: 2.0, bands: { 63: 1.2, 125: 1.2, 2000: .7, 4000: .4 }, pre: .03, build: .04, hicut: 3600, early: [[36, .2, -.8, 2600], [53, .18, .8, 2400], [88, .12, .1, 1900]] },                          // f3 f4
  fire:      { t60: 1.2, bands: { 63: 1.4, 125: 1.2, 2000: .5, 4000: .3 }, pre: .012, hicut: 2600 },                                                                                                            // f5
  ash:       { t60: .45, tail: .5, bands: { 2000: .5, 4000: .3 }, pre: .01, hicut: 2600 },                                                                                                                      // h1
  snow:      { t60: .35, tail: .22, bands: { 1000: .6, 4000: .3 }, pre: .012, hicut: 2400 },                                                                                                                    // g1 g2
  hall:      { t60: 2.3, bands: { 63: .8, 125: .95, 2000: .8, 4000: .5, 8000: .25 }, pre: .038, build: .05, hicut: 4600, early: [[31, .14, -.6, 4000], [43, .12, .6, 3600], [67, .09, 0, 3000]] },                              // voice in E and G
  white:     { t60: 5.0, bands: { 63: .5, 125: .7, 4000: .7, 8000: .4 }, pre: .03, build: .25, hicut: 5200 },                                                                                                   // h4: falling away into white
};
const cache = {}; let k = 0;
export function getRoom(name) { if (!ROOMS[name]) throw new Error('no room ' + name); return cache[name] ??= room({ seed: 40 + Object.keys(ROOMS).indexOf(name), ...ROOMS[name] }); }

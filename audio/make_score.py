"""Writes audio/score.json: the whole song (melody, lyrics, chords, sections) on the film's bar grid.  Python stdlib only."""
import json
BAR_E = 8
LINES = [  # id, start bar, voice, text (kana/kanji), english, notes "mora:pitch:eighths" ; r = rest, Q = closed consonant (small tsu), N = moraic n, > = glide
 ('b1', 4, 'lead', 'ひだまりのなか ねむるひと', 'In a patch of sun, someone is asleep', 'hi:A3:1 da:C4:1 ma:C4:1 ri:A3:1 no:G3:1 na:A3:1 ka:F3:2 ne:E3:1 mu:G3:1 ru:G3:1 hi:A3:1 to:G3:4'),
 ('b2', 6, 'lead', 'はしゃぐこえが さかみちを', 'Laughing voices run down the hill', 'ha:A3:1 sha:A3:1 gu:C4:1 ko:D4:2 e:C4:1 ga:A3:2 sa:Bb3:1 ka:A3:1 mi:G3:1 chi:F3:1 o:G3:4'),
 ('b3', 8, 'lead', 'だまれといえば またわらう', 'Tell it to shut up and it laughs again', 'da:C4:1 ma:C4:1 re:C4:1 to:A3:1 i:A3:1 e:G3:1 ba:A3:1 r::1 ma:G3:1 ta:A3:1 wa:C4:1 ra:D4:1 u:C4:3 r::1'),
 ('b4', 10, 'lead', 'さいきょうだって うそぶいて', '"I am the strongest," he boasts', 'sa:D4:1 i:E4:1 kyo:F4:2 da:E4:1 Q:E4:1 te:D4:1 r::1 u:C4:1 so:D4:1 bu:C4:1 i:Bb3:1 te:D4:3 r::1'),
 ('b5', 12, 'lead', 'ざんぎょうぎらいの あのひとも', 'Even the one who hates overtime', 'za:Bb3:1 N:Bb3:1 gyo:D4:2 gi:D4:1 ra:C4:1 i:Bb3:1 no:A3:1 a:C4:1 no:C4:1 hi:E4:1 to:D4:1 mo:C4:4'),
 ('b6', 14, 'lead', 'おなじゆうひを みていた', 'Was watching the same sunset', 'o:D4:1 na:D4:1 ji:F4:1 yu:F4:2 hi:E4:1 o:D4:1 r::1 mi:E4:1 te:D4:1 i:C4:1 ta:C4>C#4:5'),
 ('c1', 16, 'lead', 'そらがとじる よるがくる', 'The sky closes, the night comes', 'so:D4:1 ra:D4:1 ga:D4:1 to:E4:1 ji:F4:1 ru:E4:3 yo:D4:1 ru:D4:1 ga:D4:1 ku:E4:1 ru:C4:4'),
 ('c2', 18, 'lead', 'だれかがぼくを みている', 'Something is watching me', 'da:D4:1 re:D4:1 ka:F4:1 ga:F4:1 bo:G4:1 ku:F4:1 o:E4:2 mi:E4:1 te:E4:1 i:F4:1 ru:E4:5'),
 ('c3', 20, 'lead', 'こわくても このてを', 'Even afraid, these hands', 'ko:D4:1 wa:E4:1 ku:F4:1 te:G4:1 mo:G4:4 ko:F4:1 no:G4:1 te:A4:2 o:A4:4'),
 ('c4', 22, 'lead', 'にぎれ', 'Clench them', 'r::4 ni:G4:2 gi:G4:2 re:A4:8'),
 ('d1', 24, 'lead', 'かげよほえろ きばをむけ', 'Shadow, howl, bare your fangs', 'ka:D4:1 ge:F4:1 yo:A4:2 ho:G4:1 e:F4:1 ro:G4:2 ki:E4:1 ba:G4:1 o:G4:1 mu:A4:1 ke:G4:4'),
 ('d2', 26, 'lead', 'いたみごとうて ゆめのくぎ', 'Drive it in with the pain, the nail of a dream', 'i:E4:1 ta:E4:1 mi:A4:1 go:A4:1 to:G4:1 u:E4:1 te:G4:2 yu:F4:1 me:E4:1 no:D4:1 ku:E4:1 gi:F4:4'),
 ('d3', 28, 'lead', 'てをたたけば いれかわる', 'Clap your hands and we trade places', 'te:D4:1 o:F4:1 ta:A4:1 ta:A4:1 ke:G4:1 ba:F4:1 r::2 i:E4:1 re:G4:1 ka:G4:1 wa:A4:1 ru:A4:4'),
 ('d4', 30, 'lead', 'くろくひかれ このこぶし', 'Shine black, this fist', 'ku:A4:1 ro:A4:1 ku:G4:1 hi:A4:1 ka:Bb4:1 re:A4:3 ko:F4:1 no:G4:1 ko:A4:1 bu:A4:1 shi:D5:4'),
 ('d5', 32, 'lead', 'あおとあかが とけあって', 'Blue and red melt into one', 'a:D4:1 o:F4:1 to:F4:1 a:A4:1 ka:A4:1 ga:G4:3 to:G4:1 ke:G4:1 a:A4:1 Q:A4:1 te:C5:4'),
 ('d6', 34, 'lead', 'むげんのそこへ おちていけ', 'Fall to the bottom of infinity', 'mu:A4:1 ge:A4:1 N:A4:1 no:G4:1 so:F4:1 ko:E4:1 e:D4:2 o:F4:1 chi:E4:1 te:D4:1 i:C4:1 ke:D4:4'),
 ('e1', 36, 'lead', 'はこにきえた あおいそら', 'The blue sky that vanished into a box', 'ha:D4:1 ko:D4:1 ni:F4:2 ki:E4:1 e:D4:1 ta:C4:2 a:A3:1 o:C4:1 i:C4:2 so:D4:1 ra:C4:3'),
 ('e2', 38, 'lead', 'うみがみたいと いったひと', 'The one who said he wanted to see the sea', 'u:Bb3:1 mi:Bb3:1 ga:D4:1 mi:D4:1 ta:F4:1 i:E4:1 to:D4:2 i:C4:1 Q:C4:1 ta:C4:1 hi:A3:2 to:A3:3'),
 ('e3', 40, 'lead', 'あとはたのむと わらうから', '"The rest is yours," he said, and smiled', 'a:Bb3:1 to:Bb3:1 wa:Eb4:1 ta:Eb4:1 no:F4:1 mu:Eb4:1 to:D4:2 wa:D4:1 ra:D4:1 u:F4:2 ka:Eb4:1 ra:D4:3'),
 ('e4', 42, 'lead', 'みっつのかげが ひとつになる', 'Three shadows become one', 'mi:Bb3:1 Q:Bb3:1 tsu:Bb3:1 no:D4:1 ka:G4:1 ge:F4:1 ga:E4:2 hi:E4:1 to:E4:1 tsu:E4:1 ni:F4:1 na:E4:1 ru:C#4:3'),
 ('e5', 44, 'lead', 'あめよ なかないで', 'Rain, please do not cry', 'a:F4:2 me:G4:2 yo:A4:8 r::4 na:F4:1 ka:F4:1 na:A4:2 i:G4:2 de:F4>E4:10'),
 ('f1', 48, 'choir', 'のろい', 'Curse', 'no:D3:4 ro:D3:4 i:Eb3:8'),
 ('f2', 50, 'choir', 'ひらけ ひらけ', 'Open. Open.', 'hi:D3:2 ra:F3:2 ke:Eb3:4 hi:D3:2 ra:G3:2 ke:F3:4'),
 ('f3', 52, 'choir', 'きれ きれ きれ きれ きれ', 'Cut, cut, cut', 'ki:D4:1 re:D4:1 r::2 ki:D4:1 re:D4:1 r::2 ki:D4:1 re:D4:1 ki:Eb4:1 re:Eb4:1 ki:F4:1 re:F4:1 r::2'),
 ('f4', 54, 'choir', 'まわれ まわれ', 'Turn. Turn.', 'ma:D3:2 wa:F3:2 re:Eb3:4 ma:D3:2 wa:G3:2 re:A3:4'),
 ('f5', 56, 'whisper', 'ひらけ', 'Open', 'r::4 hi:D3:1 ra:D3:1 ke:D3:2 r::8'),
 ('h1', 58, 'lead', 'いかりでもいい もえてゆけ', 'Wrath will do, burn on', 'i:E4:1 ka:G4:1 ri:B4:1 de:B4:1 mo:A4:1 i:G4:1 i:A4:2 mo:F#4:1 e:A4:1 te:A4:1 yu:B4:1 ke:A4:4'),
 ('h2', 60, 'lead', 'きみがくれた いたみごと', 'With all the pain you gave me', 'ki:F#4:1 mi:F#4:1 ga:B4:1 ku:B4:1 re:A4:1 ta:F#4:3 i:G4:1 ta:F#4:1 mi:E4:1 go:F#4:1 to:G4:4'),
 ('h3', 62, 'lead', 'くろいはなが さきみだれ', 'Black flowers bloom and bloom', 'ku:E4:1 ro:G4:1 i:B4:2 ha:A4:1 na:G4:1 ga:A4:2 sa:F#4:1 ki:A4:1 mi:A4:1 da:B4:1 re:D5:4'),
 ('h4', 64, 'lead', 'のろいよ いまとけろ', 'Curse, come undone', 'no:B4:1 ro:B4:1 i:A4:1 yo:B4:5 i:G4:1 ma:F#4:1 to:E4:1 ke:F#4:1 ro:E4:4'),
 ('g2', 68, 'lead', 'ゆきがとけて てをつなぐ', 'The snow melts, our hands join', 'yu:A3:1 ki:B3:1 ga:C4:2 to:B3:1 ke:A3:1 te:G3:2 te:A3:1 o:B3:1 tsu:D4:2 na:B3:1 gu:A3:3 r::8'),
 ('g3', 71, 'lead', 'のろいもいわいも おなじくち', 'Curse and blessing come from the same mouth', 'no:E4:1 ro:G4:1 i:E4:1 mo:D4:1 i:E4:1 wa:G4:1 i:A4:1 mo:G4:1 o:F#4:1 na:F#4:1 ji:G4:1 ku:A4:1 chi:A4:4'),
 ('g4', 73, 'lead', 'いきて', 'Live', 'r::4 i:D4:2 ki:E4:2 te:G4:10 r::6'),
]
CHORDS = ['Dm','Dm','Dm','Dm'] + ['F','C/E','Dm7','Bb','F','C/E','Dm7','Bb','Gm7','Am7','Bb','A7'] + \
 ['Dm','Dm/C','Bb','A7','Gm','Dm/F','Eb','A7'] + ['Bb','C','Am','Dm']*3 + \
 ['Bbmaj7','F/A','Gm7','F','Eb','Bb/D','Gm7','A7sus4','Bb','C','Dm','A7'] + \
 ['D5','D5','D5','Eb5','D5','Eb5','F5','Eb5','N','A7b9'] + ['C','D','Bm','Em','C','D','B7','Em'] + \
 ['Cmaj7','G/B','Am7','D/F#','G','C','D','Cmaj7','G','G']
assert len(CHORDS) == 76
SECTIONS = [('A', 0, 'fear: black, the wheel'), ('B', 4, 'warmth and comedy'), ('C', 16, 'dread'), ('D', 24, 'action chorus'), ('E', 36, 'sadness, half-time'),
            ('F', 48, 'Sukuna: ritual, heavy'), ('F5', 56, 'stop-time, fire'), ('H', 58, 'wrath chorus, up a tone'), ('G', 66, 'snow, forgiveness')]
out = []
for i, bar, voice, ja, en, notes in LINES:
    t = 0; ns = []
    for tok in notes.split():
        m, p, d = tok.split(':'); d = float(d)
        if m != 'r': ns.append(dict(mora=m, pitch=p, t=bar * 8 + t, d=d))
        t += d
    assert abs(t % 8) < 1e-9, (i, t)
    out.append(dict(id=i, bar=bar, voice=voice, ja=ja, en=en, beats8=t, notes=ns))
json.dump(dict(bpm=160, eighth_s=.1875, bar_s=1.5, n_bars=76, seconds=114.0, sections=SECTIONS, chords=CHORDS, lines=out), open(__file__.replace('make_score.py', 'score.json'), 'w'), ensure_ascii=False, indent=1)
print(len(out), 'lines', sum(len(l['notes']) for l in out), 'notes')

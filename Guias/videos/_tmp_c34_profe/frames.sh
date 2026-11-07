cd "/c/Users/Jason Leon/Desktop/MaterialEstudioDS/Guias/videos"
while read n ts; do
  f=$(ls C34/C34_${n}_*.mp4)
  for t in $ts; do ffmpeg -y -loglevel error -ss $t -i "$f" -frames:v 1 -vf scale=960:-1 "_tmp_c34_profe/frames_mp4/C34_${n}_t${t}.png"; done
done <<LIST
01 60 210
02 150 178 205
03 110 128 185
04 100 108 135
05 80 197 222
06 102 178 200
07 135 192 222
08 140 166
09 75 120 192
10 186 226
11 85
LIST
ls _tmp_c34_profe/frames_mp4 | wc -l

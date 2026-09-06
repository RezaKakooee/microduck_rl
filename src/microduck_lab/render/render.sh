#!/bin/bash
# One renderer for every task. Six near-identical scripts used to do this, each
# repeating the same cd, the same MUJOCO_GL and the same uv invocation, with the
# repo path hard-coded in all six.
#
# Offscreen rendering needs EGL, so this has to run on a GPU:
#   bash src/microduck_lab/render/render.sh <target>
#
# Targets:
#   film        the dispenser love story          -> videos/love_story/
#   film-v2     love story v2                     -> videos/love_story/
#   film-dry    v2, no GPU, report only
#   story       the legacy two-duck story         -> videos/love_story/story/
#   clips       one clip per pretrained policy    -> videos/policy_demos/
#   ice         four ice clips                    -> videos/ice_experts/
#   pickup      three pick-up layouts             -> videos/pick_up/
#   kick        three ball-kick layouts           -> videos/kick_ball/
#
# Output goes under videos/<task>/, which is a symlink into local_storage.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../../.." || exit 1
export MUJOCO_GL=egl
export OPENBLAS_NUM_THREADS=1

V=${OUT:-videos}
# Defaults to a `microduck` checkout beside this one; override with POLICIES.
P=${POLICIES:-"$(cd .. && pwd)/microduck/policies"}
R="uv run --with imageio --with imageio-ffmpeg python -m"
PY=.venv/bin/python

out() { mkdir -p "$V/$1"; echo "$V/$1"; }

# A 960-wide review copy and a 4x3 contact sheet beside any rendered film.
small_and_sheet() {
  local mp4=$1 base=${1%.mp4}
  local ff; ff=$($PY -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())")
  "$ff" -y -loglevel error -i "$mp4" -vf scale=960:-2 -crf 30 -pix_fmt yuv420p "${base}_small.mp4"
  uv run --with imageio --with imageio-ffmpeg --with pillow \
     python -m microduck_lab.tools.contact_sheet "$mp4" -o "${base}_sheet.jpg"
  ls -la "${base}_small.mp4" "${base}_sheet.jpg"
}

case "${1:-film}" in

film)
  D=$(out love_story)
  $R microduck_lab.tasks.love_story.dispenser \
     --video "$D/duck_love_story_physics.mp4" --report "$D/duck_love_story_physics.json" ;;

film-v2)
  D=$(out love_story)
  $R microduck_lab.tasks.love_story.v2 \
     --video "$D/duck_love_story_v2.mp4" --report "$D/duck_love_story_v2.json"
  # A 100 MB master is awkward to look at; make the review copy and the sheet
  # in the same step so they cannot drift from the film.
  small_and_sheet "$D/duck_love_story_v2.mp4" ;;

film-dry)
  D=$(out love_story)
  uv run python -m microduck_lab.tasks.love_story.v2 --dry --trace \
     --report "$D/duck_love_story_v2.json" ;;

film-legacy)
  D=$(out love_story)
  $R microduck_lab.tasks.love_story.original --video "$D/duck_love_story_legacy.mp4" ;;

film-posed)
  D=$(out love_story)
  $R microduck_lab.film.film --out "$D/duckfilm_posed.mp4" --width 1280 --height 720 ;;

film-puppet)
  D=$(out love_story)
  $R microduck_lab.film.physics --out "$D/duckfilm_puppet.mp4" --width 1280 --height 720 ;;

story)
  D=$(out love_story/story)
  $PY -m microduck_lab.film.legacy_story_two_ducks --check
  $PY -m microduck_lab.film.legacy_story_two_ducks --out "$D" "${@:2}"
  FF=$($PY -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())")
  "$FF" -y -loglevel error -i "$D/story.mp4" -vf scale=960:-2 -crf 30 \
        -pix_fmt yuv420p "$D/story_small.mp4"
  "$FF" -y -loglevel error -i "$D/story.mp4" \
        -vf "select='not(mod(n,84))',scale=320:-1,tile=6x4" -frames:v 1 \
        "$D/story_sheet.png" ;;

clips)
  D=$(out policy_demos)
  R7="$R microduck_lab.tools.headless_rollout --new-cmd-obs"
  echo "### 1 walk"
  $R7 --walking "$P/alpha_walking.onnx" --lin-vel-x 0.3 --seconds 12 --video "$D/1_walk.mp4" 2>/dev/null | grep -E "travelled|body speed|fell"
  echo "### 2 turn in place"
  $R7 --walking "$P/alpha_walking.onnx" --ang-vel-z 1.0 --seconds 10 --video "$D/2_turn.mp4" 2>/dev/null | grep -E "yaw rate|fell"
  echo "### 3 roulade"
  $R7 --walking "$P/alpha_walking.onnx" --roulade "$P/roulade.onnx" --do roulade@2 --seconds 8 --video "$D/3_roulade.mp4" 2>/dev/null | grep -E "travelled|yaw rate|fell"
  echo "### 4 ground pick"
  $R7 --walking "$P/alpha_walking.onnx" --ground-pick "$P/alpha_ground_pick.onnx" --do ground_pick@2 --seconds 10 --video "$D/4_ground_pick.mp4" 2>/dev/null | grep -E "trunk height|fell"
  echo "### 5 sit then stand"
  $R7 --sitstand "$P/alpha_sitstand.onnx" --do sit@2 --do sit@7 --seconds 12 --video "$D/5_sitstand.mp4" 2>/dev/null | grep -E "trunk height|fell"
  echo "### 6 ball kick"
  $R7 --walking "$P/alpha_walking.onnx" --kick-right "$P/ball_kick_right.onnx" --do kick_right@2 --seconds 8 --video "$D/6_kick.mp4" 2>/dev/null | grep -E "travelled|fell"
  echo "### 7 rollers"
  $R7 --walking "$P/roller.onnx" --roller --lin-vel-x 0.4 --seconds 12 --video "$D/7_rollers.mp4" 2>/dev/null | grep -E "travelled|body speed|fell" ;;

ice)
  D=$(out ice_experts)
  ICE=src/microduck_lab/models/scene_ice.xml
  RI="$R microduck_lab.tools.headless_rollout --new-cmd-obs --xml $ICE"
  echo "### A baseline: the WALKING policy on ice"
  $RI --walking "$P/alpha_walking.onnx" --foot-friction 0.08 --lin-vel-x 0.3 --seconds 12 --video "$D/ice_a_walker_fails.mp4" 2>/dev/null | grep -E "body speed|fell:"
  echo "### B the ice policy skating (mu 0.12)"
  $RI --walking ice_v2_iter4000.onnx --foot-friction 0.12 --lin-vel-x 0.3 --seconds 18 --video "$D/ice_b_skating.mp4" 2>/dev/null | grep -E "body speed|fell:"
  echo "### C same policy on much slicker ice (mu 0.05)"
  $RI --walking ice_v2_iter4000.onnx --foot-friction 0.05 --lin-vel-x 0.3 --seconds 18 --video "$D/ice_c_slicker.mp4" 2>/dev/null | grep -E "body speed|fell:"
  echo "### D the collapsed final policy -- stands still"
  $RI --walking ice_v2_iter7999.onnx --foot-friction 0.12 --lin-vel-x 0.3 --seconds 12 --video "$D/ice_d_collapsed.mp4" 2>/dev/null | grep -E "body speed|fell:" ;;

pickup)
  D=$(out pick_up)
  G="$R microduck_lab.tasks.objects.pick_up --walking $P/alpha_walking.onnx --ground-pick $P/alpha_ground_pick.onnx"
  M="beak down|GOT IT|standing up|closest|lifted|carried|task:"
  echo "### pick A: cube ahead-left"
  $G --cube 1.0 0.5 --seconds 70 --cam-distance 0.7 --video "$D/pick_a_left.mp4" 2>/dev/null | grep -E "$M"
  echo "### pick B: cube ahead-right"
  $G --cube 1.0 -0.5 --seconds 70 --cam-distance 0.7 --video "$D/pick_b_right.mp4" 2>/dev/null | grep -E "$M"
  echo "### pick C: cube BEHIND the duck"
  $G --cube -0.9 0.4 --seconds 90 --cam-distance 0.7 --video "$D/pick_c_behind.mp4" 2>/dev/null | grep -E "$M" ;;

kick)
  D=$(out kick_ball)
  K="$R microduck_lab.tasks.objects.kick_ball --walking $P/alpha_walking.onnx --kick-left $P/ball_kick_left.onnx --kick-right $P/ball_kick_right.onnx"
  M="aiming|close|settled in|kicked|ball moved|task:|never"
  echo "### task A: ball ahead-left"
  $K --ball 1.2 0.6 --seconds 70 --video "$D/task_a_left.mp4" 2>/dev/null | grep -E "$M"
  echo "### task B: ball ahead-right"
  $K --ball 1.2 -0.6 --seconds 70 --video "$D/task_b_right.mp4" 2>/dev/null | grep -E "$M"
  echo "### task C: ball BEHIND the duck"
  $K --ball -1.0 0.5 --seconds 90 --video "$D/task_c_behind.mp4" 2>/dev/null | grep -E "$M" ;;

*)
  echo "unknown target: ${1:-}"; sed -n "10,20p" "${BASH_SOURCE[0]}"; exit 1 ;;
esac

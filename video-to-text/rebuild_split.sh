echo '# Build MSVD dataset annotations:'
python2 build_msvd_annotation.py
echo '# Prepare dataset split'
python2 prepare_split.py
echo '# Build vocabulary'
python2 vocab.py
echo '# Convert each caption to token index list'
python2 prepare_caption.py
echo '# Prepare ground-truth'
python2 prepare_gt.py

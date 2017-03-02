import pkl


vid_to_labels = pkl.load(open("/data/uts700/linchao/yt8m/YT/data/vid_info/train_vid_to_labels.pkl"))
label_to_vids = {}
cnt = 0
for vid, labels in vid_to_labels.iteritems():
    for l in labels:
        c = label_to_vids.get(l, [])
        c.append(vid)
        label_to_vids[l] = c
        cnt += 1
# pkl.dump(label_to_vids, open("/data/uts700/linchao/yt8m/YT/data/vid_info/train_label_to_vids.pkl", "w"))

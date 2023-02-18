# Graph-coloring-problem

## Dataset說明
* 相同session: 相同組user
* 相同round: 同一個graph -> 某個時期的狀態
* node feature: id, color, score, history(each action %), neighbor_num
* actionType = ground truth

---
### step 1
* session + user_id + round 可以決定一個node (primary key)
* 把action == 7刪除 (代表結束)，用session、round、id重新排序
* 整理與合併 dataframe

### step 2
分成graph feature和graph sturcture
* graph feature: session, id, round, size, color, score, num_of_neighbor, hist_color, hist_neighbor, hist_skip
* graph structure: session, id, round, size, linked

### step 3
分成color change視角的dataframe 和 neighbor change視角的dataframe
1. color change
    (不能把沒選color的node刪除 -> 他們也會提供其他user資訊!)
    * 只能都留著，在訓練的時候只看new_color != 996 的 node

2. neighbor change
    同理上面color change
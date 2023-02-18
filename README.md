# Graph-coloring-problem
 
## Model說明
* 透過讓多名受試者人來解決著色問題，訓練一個可以預測受試者行為的model  
    注意：與一般的著色問題不同，受試者除了更改顏色外，還可以選擇更改相鄰的其中一位鄰居
    
* model架構分為兩層
    1. 第一層預測受試者的選擇 (color、neighbor、nothing)
    2. 第二層分為兩種可能，若是選擇color則進一步預測更換的color為何；若是選擇neighbor則預測會刪除哪位既有鄰居、新增哪位新鄰居
        (目前只時做了預測color，neighbor部分尚未實作)


## Dataset說明
* 相同session: 相同組user
* 相同round: 同一個graph -> 某個時期的狀態
* node feature: id, color, score, history(each action %), neighbor_num
* actionType = ground truth


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

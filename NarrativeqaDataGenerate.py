import pandas as pd
import glob

# 由于dataset文件太大无法上传至github，此代码仅供演示
# 加载一个 parquet 文件
parquet_path = "/content/drive/MyDrive/Narrativeqa/narrativeqa/data/train-00000-of-00024.parquet"
data = pd.read_parquet(parquet_path)

# 查看字段信息
print(data.columns)
print(data.head())



# 生成一千条queries以及取回相应的文本来源

parquet_dir = "/content/drive/MyDrive/Narrativeqa/narrativeqa/data/"
save_path = "/content/drive/MyDrive/Narrativeqa/1wq_1.5wa_deduplication/"
context_file = save_path + "context.tsv"
query_file = save_path + "query.tsv"


parquet_files = glob.glob(parquet_dir + "train-*.parquet")
all_data = []
for file in parquet_files:
    print(f"Loading: {file}")
    data = pd.read_parquet(file)
    all_data.append(data)

full_data = pd.concat(all_data, ignore_index=True)
print(f"Total records in full_data: {len(full_data)}")


# 每个 document.id 分组后，随机保留一条记录
unique_data = (
    full_data
    .groupby(lambda idx: full_data.loc[idx, 'document']['id'], group_keys=False)
    .apply(lambda g: g.sample(n=1, random_state=42))
    .reset_index(drop=True)
)
print(f"Unique documents after group sampling (1 Q&A per doc): {len(unique_data)}")


sample_size = 10000
if len(unique_data) > sample_size:
    main_data = unique_data.sample(n=sample_size, random_state=42).reset_index(drop=True)
else:
    main_data = unique_data  # 如果文档数不够，就使用全部
print(f"Final main_data size: {len(main_data)}")


context_data = main_data['document'].apply(
    lambda x: {
        'id': x['id'],
        'context': x['summary']['text']
    }
)
context_df = pd.DataFrame(context_data.tolist())


query_data = main_data.apply(
    lambda x: {
        'id': x['document']['id'],
        'query': x['question']['text'],
        #存在多个答案，'; '分隔
        'answer': "; ".join([ans['text'] for ans in x['answers']])
    },
    axis=1
)
query_df = pd.DataFrame(query_data.tolist())


# 干扰数据


unique_distraction = (
    full_data
    .groupby(lambda idx: full_data.loc[idx, 'document']['id'], group_keys=False)
    .apply(lambda g: g.sample(n=1, random_state=123))
    .reset_index(drop=True)
)

used_ids = set(main_data['document'].apply(lambda x: x['id']))
unique_distraction = unique_distraction[
    ~unique_distraction['document'].apply(lambda x: x['id']).isin(used_ids)
].reset_index(drop=True)


distraction_size = 5000
if len(unique_distraction) >= distraction_size:
    distraction_data = unique_distraction.sample(n=distraction_size, random_state=123)
else:

    distraction_data = unique_distraction

distraction_data = distraction_data.reset_index(drop=True)
print(f"Sampled {len(distraction_data)} records as distraction data.")


distraction_context_data = distraction_data['document'].apply(
    lambda x: {
        'id': x['id'],
        'context': x['summary']['text']
    }
)
distraction_df = pd.DataFrame(distraction_context_data.tolist())

context_with_distraction = pd.concat([context_df, distraction_df], ignore_index=True)
context_shuffled = context_with_distraction.sample(frac=1, random_state=456).reset_index(drop=True)


context_shuffled.to_csv(context_file, sep='\t', index=False)
query_df.to_csv(query_file, sep='\t', index=False)


print(f"Context（含干扰）保存至: {context_file}, 共 {len(context_shuffled)} 条记录。")
print(f"Query 数据集保存至: {query_file}, 共 {len(query_df)} 条记录。")

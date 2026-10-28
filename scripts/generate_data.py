from data.generation import generate_distributed_dataset
import json
import random
import multiprocessing
import time
import pickle
import gzip
import os


with open('configs/pretrain.json', 'r') as f:   # 'r'表示只可读
    configs = json.load(f)



class DatasetGenerator:
    '''
    并行数据集生成类
    '''
    def __init__(self, output_dir='./datasets/pretrain', num_processes=16, total_samples:int= configs["model_configs"]["model_config1"]['num_samples'],
                 samples_per_file=configs["model_configs"]["model_config1"]['samples_per_file'], data_dir='./datasets/pretrain',
            temperature=configs["model_configs"]["model_config1"]['temperature'], val_split = configs["model_configs"]["model_config1"]['val_split']):
        self.output_dir = output_dir
        self.num_processes = num_processes
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.total_files_generated = 0
        self.start_time = None
        self.num_total_samples = total_samples
        self.samples_perfile = samples_per_file
        self.datadir = data_dir
        self.temperature = temperature
        self.val_split = val_split

    def generate_dataset(self):
        self.start_time = time.time()

        print(f"开始生成数据集")
        print(f"输出目录: {self.output_dir}")
        print(f"进程数: {self.num_processes}")
        print("-" * 50)
        num_per_process = self.num_total_samples // self.num_processes
        metadata_train = {}
        metadata_val = {}
        num_total_samples_train = 0
        num_files_train = 0
        filesample_index_mapping_train = {}
        num_total_samples_val = 0
        num_files_val = 0
        filesample_index_mapping_val = {}
        with multiprocessing.Pool(processes=self.num_processes) as pool:
            tasks = [(num_per_process, self.samples_perfile, self.datadir, self.val_split, self.temperature, i) for i in range(self.num_processes)]
            async_results = []  # 初始化任务列表

            # 主进程提交任务队列
            for task_args in tasks:
                async_result = pool.apply_async(generate_distributed_dataset, task_args)
                async_results.append(async_result)
                print(f"已提交任务 {task_args[5]}")

            print("所有任务提交完成，等待文件生成...")

            for i, async_result in enumerate(async_results):
                try:
                    num_total_samples_train += async_result.get()[0]['total_num_samples']
                    num_files_train += async_result.get()[0]['num_files']
                    filesample_index_mapping_train.update(async_result.get()[0]['filesample_index_mapping'])

                    num_total_samples_val += async_result.get()[1]['total_num_samples']
                    num_files_val += async_result.get()[1]['num_files']
                    filesample_index_mapping_val.update(async_result.get()[1]['filesample_index_mapping'])
                    print(f"✅任务 {i} 完成")

                except multiprocessing.TimeoutError:
                    print(f"⏰ 任务 {i} 超时!")
                except Exception as e:
                    print(f"❌ 任务 {i} 异常: {e}")
            metadata_train["total_num_samples"] = num_total_samples_train
            metadata_train["num_files"] = num_files_train
            metadata_train["filesample_index_mapping"] = filesample_index_mapping_train
            metadata_val["total_num_samples"] = num_total_samples_val
            metadata_val["num_files"] = num_files_val
            metadata_val["filesample_index_mapping"] = filesample_index_mapping_val

            train_dir = os.path.join(self.datadir, "train_data")
            val_dir = os.path.join(self.datadir, "val_data")
            metadata_train_name = "metadata.pkl.gz"
            metadata_train_path = os.path.join(train_dir, metadata_train_name)
            with gzip.open(metadata_train_path, 'wb') as f:
                pickle.dump(metadata_train, f)
            metadata_val_name = "metadata.pkl.gz"
            metadata_val_path = os.path.join(val_dir, metadata_val_name)
            with gzip.open(metadata_val_path, 'wb') as f:
                pickle.dump(metadata_val, f)
            self.endtime = time.time()
            print(f"任务用时{self.endtime-self.start_time:.2f}s")



    def progress_callback(self, result):
        """进度回调函数"""
        self.completed_tasks += 1

        if result['success']:
            file_count = len(result['generated_files'])
            self.total_files_generated += file_count
            elapsed = time.time() - self.start_time
            print(f"任务 {result['chunk_id']} 完成! "
                  f"生成 {file_count} 个文件, "
                  f"进度: {self.completed_tasks}/{self.total_tasks} "
                  f"({elapsed:.1f}s)")
        else:
            self.failed_tasks += 1
            print(f"任务 {result['chunk_id']} 失败!")

    def error_callback(self, error):
        """错误回调函数"""
        self.failed_tasks += 1
        print(f"任务执行出错: {error}")



if __name__ == '__main__':
    # 对于Windows，有时还需要添加这行
    multiprocessing.freeze_support()
    gendata = DatasetGenerator()
    gendata.generate_dataset()
"""按上传清单逐文件比对当前运行源码与输入，不计算摘要。"""
import json
import os
from pathlib import Path
import tarfile

ROOT=Path(os.environ["MYMIND_WORK_ROOT"])
RUN=ROOT / "runs/stage7-preparation-20261014"


def compare(archive_path, target_root, remap=False):
    matched=[]
    with tarfile.open(archive_path,"r:gz") as archive:
        for member in archive:
            if not member.isfile(): continue
            name=member.name
            if remap: name=name.replace("agentic_runs/","runs/",1)
            expected=archive.extractfile(member).read()
            assert (target_root / name).read_bytes()==expected,name
            matched.append(name)
    return matched


def main():
    source=compare(ROOT / "tmp/source-final.tar.gz",ROOT / "src")
    metadata=compare(ROOT / "tmp/stage7-final-metadata.tar.gz",ROOT,True)
    # 初次同步的配置/额度已合法演进，只比对仍固定的标准、共同池及正式导出。
    data=[]
    with tarfile.open(ROOT / "tmp/data-sync.tar.gz","r:gz") as archive:
        for member in archive:
            if member.isfile() and member.name.startswith("agentic_runs/pre-stage7-fixes-20261014/"):
                name=member.name.replace("agentic_runs/","runs/",1)
                assert (ROOT / name).read_bytes()==archive.extractfile(member).read(),name
                data.append(name)
    result=dict(status="passed",source_files=source,metadata_files=metadata,fixed_data_files=data,
        comparison="逐字节；包含未提交文件；仅清单同步",model_calls=0,api_requests=0)
    (RUN / "sync-verification.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"status":"passed","source_files":len(source),"metadata_files":len(metadata),"fixed_data_files":len(data)}))


if __name__=="__main__": main()

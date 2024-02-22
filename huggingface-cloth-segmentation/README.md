# huggingface-cloth-segmentation
- 옷 사진만 mask하는 코드
- 모델 다운로드 필요
  - https://drive.google.com/uc?id=11xTBALOeUkyuaK3l60CpkYHLTmv7k3dY
  - app.py와 동일 위치에 model 폴더 생성 후 폴더 안에 위치
    
## 수정사항
- network : 기존에서 변환된 파라미터로 변경
- options : output 기본값 제거
- single_process : 불필요 함수 및 과정 제거
- network와 options는 동일명 파일을 대체하고 process 대신 single process 사용

## 실행 명령어
```
python single_process.py --image "../test_image/0002_00.jpg" --output "../test"
```

- 설명
  ~~~
  python single_process.py --image "옷 사진.jpg" --output "이미지 저장 경로"
  ~~~

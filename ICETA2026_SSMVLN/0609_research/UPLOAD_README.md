# GitHub Upload Folder Guide

작성일: 2026-06-09

## 1. GitHub repo로 올릴 폴더

아래 폴더를 GitHub repository로 올리면 된다.

```text
/home/mj/my_research/github_upload/iceta_ssm_nav_repo
```

이 폴더에는 코드와 문서만 정리되어 있다.

포함 내용:

```text
README.md
.gitignore
scripts/
docs/
src/
```

## 2. GitHub Release 또는 별도 압축 첨부용 폴더

아래 폴더는 repo에 바로 넣기보다 GitHub Release artifact 또는 별도 압축파일로 첨부하는 것을 권장한다.

```text
/home/mj/my_research/github_upload/iceta_ssm_nav_release_artifacts
```

포함 내용:

```text
models/iceta_semantic_prior_ssm.pt
maps/iceta_semantic_prior/
results/professor_demo/iceta_semantic_prior/
```

## 3. 업로드 추천 방식

권장:

```text
GitHub repo:
  /home/mj/my_research/github_upload/iceta_ssm_nav_repo

GitHub Release artifact:
  /home/mj/my_research/github_upload/iceta_ssm_nav_release_artifacts
```

## 4. 바로 실행할 명령

```bash
cd /home/mj/my_research/github_upload/iceta_ssm_nav_repo
git init
git add README.md .gitignore scripts docs src
git commit -m "Prepare ICETA SSM navigation simulation artifact"
git remote add origin https://github.com/<USER>/<REPO>.git
git branch -M main
git push -u origin main
```

## 5. 주의

- `/home/mj/my_research/ssm_nav_ws` 전체를 올리지 않는다.
- `build/`, `install/`, `log/`, `datasets/`, 전체 `results/`, 전체 `models/`는 올리지 않는다.
- 현재 정리된 repo 폴더만 올리면 깔끔하다.


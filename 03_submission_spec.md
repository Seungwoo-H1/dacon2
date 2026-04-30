# 제출물 명세(리더보드/논문/코드)

## A. 리더보드 제출
- 제출 위치: 데이콘 대회 제출 탭
- 기본 파일 참조: `ch2026_submission_sample.csv`
- 제한: 1일 최대 3회 제출
- 점수: Average Log-Loss (Public 44%, Private 100%)

## B. 논문 제출(ICTC 2026)
- 제출 시스템: ICTC 2026 EDAS
- 트랙: ICTC Workshop on ETRI Human Understanding AI Paper Challenge (IWETRIAI)
- 양식: Standard IEEE conference templates, 6-page full paper
- 마감: 06.26
- 채택 결과: 09.01

## C. 코드/모델설명서 제출(논문 채택팀)
- 제출 대상: 논문 채택팀
- 제출처: `dacon@dacon.io` (팀명 포함)
- 마감: 09.01
- 코드 필수 조건:
  - `/data` 경로 포함
  - 확장자: `.R`, `.rmd`, `.py`, `.ipynb`
  - 인코딩: UTF-8
  - 실행 가능 상태(의존성 포함)
  - OS 및 라이브러리 버전 명시
  - 사전학습 모델 출처/링크 명시
  - Private LB 재현 가능
- 모델 설명서: 자유 양식

## D. 최종 평가/수상 프로세스
- 리더보드 제출 + 논문 제출 + 재현성 검증 통과한 팀만 종합평가
- 재현성 통과 논문 대상 평가위원회 종합평가
- 상위 1~5위: 10.15 발표/시상식 참여 필요

## E. 실무 제출 패키지 체크리스트
- [ ] `submission.csv` (샘플 형식 100% 일치)
- [ ] `paper.pdf` (IEEE 6페이지 규격)
- [ ] `src/` 재현 코드
- [ ] `requirements.txt`/`environment.yml`
- [ ] `README_repro.md` (실행 순서, 경로, 시드)
- [ ] `model_card.md` (모델/라이선스/제한사항)
- [ ] `external_data.md` (외부데이터 근거)

## 출처
- https://dacon.io/competitions/official/236690/overview/description
- https://dacon.io/competitions/official/236690/overview/evaluation
- https://dacon.io/competitions/official/236690/overview/rules

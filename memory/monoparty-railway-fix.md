# monoparty Railway 멀티플레이 테스트 결과

## 발견된 문제

1. **tsconfig.json: module=commonjs vs package.json: type=module 충돌**
   - `package.json`에 `"type": "module"`이 설정되어 있지만 `tsconfig.json`이 `module: "commonjs"`로 CJS 형식으로 컴파일
   - `dist/index.js`가 `exports`를 사용해 ES 모듈 환경에서 실행 불가
   - **수정**: `module: "NodeNext"`, `moduleResolution: "NodeNext"`, 모든 import에 `.js` 확장자 추가

2. **Railway URL에 불필요한 포트 포함**
   - `VITE_GAME_SERVER_URL=wss://monoparty-production.up.railway.app:8080`
   - Railway Public Domain은 HTTPS(포트 443)에서 작동, `:8080` 포함 시 연결 불가
   - **수정**: `wss://monoparty-production.up.railway.app` (포트 제거)

3. **Railway 빌드 실패: `cd` 명령어 불가**
   - `railway.json`에서 `cd game-server && npm ci` 사용
   - Railway Nixpacks에서 `cd` 명령어 미지원
   - **수정**: Railway 대시보드에서 직접 Builder 설정

## 수정 파일

| 파일 | 변경 내용 |
|------|-----------|
| `game-server/tsconfig.json` | module: "NodeNext", moduleResolution: "NodeNext" |
| `game-server/src/index.ts` | import에 `.js` 확장자 추가 (2개) |
| `game-server/src/room-manager.ts` | import에 `.js` 확장자 추가 (2개) |
| `game-server/src/room.ts` | import에 `.js` 확장자 추가 (2개) |
| `Dockerfile` | game-server 경로 수정 |
| `frontend/.env` | `:8080` 제거 |
| `railway.json` | `cd` 제거, `npm install` 사용 |

## Railway 대시보드 설정 필요

1. **Settings → Builder**
   - Root Directory: `game-server`
   - Build Command: `npm install && npm run build`
   - Start Command: `npm start`

2. **Settings → Environment Variables**
   - `PORT` = `8080`

## Vercel 환경변수 설정 필요

- `VITE_GAME_SERVER_URL` = `wss://monoparty-production.up.railway.app`

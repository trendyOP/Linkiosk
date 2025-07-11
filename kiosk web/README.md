1. 필수 요구사항 (Prerequisites)

Node.js (18 LTS 이상) 설치
- macOS: brew install node
- Windows: https://nodejs.org LTS MSI → 설치 시 Add to PATH 체크

npm(Node.js 설치 시 자동 포함)

2. 개발 서버 실행 (Development)

# 1) 프로젝트 폴더로 이동
cd <프로젝트_루트>

# 2) 의존성 설치
npm install

# 3) 개발 서버 시작
# CRA  : npm start         # 브라우저 자동 열림 → http://localhost:3000
# Vite : npm run dev       # 콘솔에 표시된 URL 접근 → http://localhost:5173

3. 프로젝트 루트에서 실행
압축 해제 후

cd <압축해제_폴더>
npm install     # node_modules 재생성
npm start       # 개발 서버
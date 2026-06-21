# Supabase Lobby Schema — 실행해서 테이블 생성

-- 1. lobby_rooms 테이블 생성
create table lobby_rooms (
  id bigint primary key generated always as identity,
  peer_id text unique not null,
  host_nickname text not null,
  host_user_id text not null,
  members jsonb not null default '[]'::jsonb,
  status text not null default 'waiting' check (status in ('waiting', 'playing', 'full')),
  game_started boolean not null default false,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

-- 2. 인덱스
create index idx_lobby_rooms_created_at on lobby_rooms(created_at desc);

-- 3. RLS 활성화
alter table lobby_rooms enable row level security;

-- 4. 읽기/쓰기 정책 (anon key로 접근)
create policy "Anyone can read rooms"
  on lobby_rooms for select using (true);

create policy "Anyone can manage rooms"
  on lobby_rooms for all
  using (true) with check (true);

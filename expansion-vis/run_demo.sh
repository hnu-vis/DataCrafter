screen -S backend -dm bash -c "(cd backend && python manager.py run)"
screen -S frontend -dm bash -c "(cd frontend && npm run dev)"

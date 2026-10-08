import React, { useEffect, useState } from 'react';
import {
  View, Text, FlatList, TextInput, TouchableOpacity, ScrollView,
  ActivityIndicator, StyleSheet, Platform, StatusBar,
} from 'react-native';

const API_URL = 'https://cricket-analytics-s1m8.onrender.com';

const C = {
  bg: '#0b1d12', card: '#14301f', accent: '#2ecc71',
  text: '#f1f5f2', muted: '#8fa89a', bad: '#e74c3c',
};

// ---------- API helpers ----------
function friendly(e) {
  if (e.name === 'AbortError') return 'The server took too long to respond. Please try again.';
  if (e.message === 'Network request failed') return 'Cannot reach the server. Check your internet connection and API_URL.';
  return e.message;
}

async function api(path, options) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 60000);
  try {
    const res = await fetch(API_URL + path, { ...options, signal: controller.signal });
    if (!res.ok) {
      let msg = 'Request failed (' + res.status + ')';
      try {
        const j = await res.json();
        if (typeof j.detail === 'string') msg = j.detail;
      } catch (_) {}
      throw new Error(msg);
    }
    return await res.json();
  } finally {
    clearTimeout(timer);
  }
}

function useApi(path) {
  const [state, setState] = useState({ data: null, loading: true, error: null });
  useEffect(() => {
    let alive = true;
    setState({ data: null, loading: true, error: null });
    api(path)
      .then((data) => alive && setState({ data, loading: false, error: null }))
      .catch((e) => alive && setState({ data: null, loading: false, error: friendly(e) }));
    return () => { alive = false; };
  }, [path]);
  return state;
}

function useDebounced(value, ms = 500) {
  const [v, setV] = useState(value);
  useEffect(() => {
    const t = setTimeout(() => setV(value), ms);
    return () => clearTimeout(t);
  }, [value, ms]);
  return v;
}

// ---------- Shared pieces ----------
function Status({ loading, error }) {
  if (loading) {
    return (
      <View style={s.center}>
        <ActivityIndicator color={C.accent} size="large" />
        <Text style={[s.muted, { marginTop: 12, textAlign: 'center' }]}>
          Loading… the free server can take up to a minute to wake up.
        </Text>
      </View>
    );
  }
  if (error) {
    return (
      <View style={s.center}>
        <Text style={s.error}>{error}</Text>
      </View>
    );
  }
  return null;
}

function Chips({ options, value, onChange }) {
  return (
    <View style={[s.row, { paddingHorizontal: 12, paddingTop: 10 }]}>
      {options.map((o) => (
        <TouchableOpacity
          key={o}
          style={[s.chip, value === o && s.chipOn]}
          onPress={() => onChange(o)}
        >
          <Text style={[s.chipText, value === o && { color: '#06210f' }]}>{o}</Text>
        </TouchableOpacity>
      ))}
    </View>
  );
}

function BarRow({ label, value, max }) {
  return (
    <View style={{ marginBottom: 10 }}>
      <View style={[s.row, { justifyContent: 'space-between' }]}>
        <Text style={s.text}>{label}</Text>
        <Text style={s.muted}>{value}</Text>
      </View>
      <View style={s.track}>
        <View style={[s.fill, { width: `${Math.max((value / max) * 100, 2)}%` }]} />
      </View>
    </View>
  );
}

function Stat({ label, value }) {
  return (
    <View style={s.statCard}>
      <Text style={s.statValue}>{value}</Text>
      <Text style={s.muted}>{label}</Text>
    </View>
  );
}

// ---------- Player rankings ----------
function PlayerList({ path, nameKey, impactKey, line }) {
  const { data, loading, error } = useApi(path);
  if (loading || error) return <Status loading={loading} error={error} />;
  return (
    <FlatList
      data={data}
      keyExtractor={(item) => item[nameKey]}
      contentContainerStyle={{ padding: 12 }}
      renderItem={({ item, index }) => (
        <View style={s.card}>
          <Text style={s.rank}>{index + 1}</Text>
          <View style={{ flex: 1 }}>
            <Text style={s.name}>{item[nameKey]}</Text>
            <Text style={s.muted}>{line(item)}</Text>
          </View>
          <Text style={s.impact}>{Math.round(item[impactKey])}</Text>
        </View>
      )}
    />
  );
}

// ---------- Overview ----------
function Overview() {
  const [fmt, setFmt] = useState('IPL');
  const summary = useApi('/overview?format=' + fmt);
  const wins = useApi('/overview/team-wins?format=' + fmt + '&limit=10');
  const seasons = useApi('/overview/matches-per-season?format=' + fmt);

  const loading = summary.loading || wins.loading || seasons.loading;
  const error = summary.error || wins.error || seasons.error;

  return (
    <View style={{ flex: 1 }}>
      <Chips options={['IPL', 'T20I', 'ODI']} value={fmt} onChange={setFmt} />
      {loading || error ? (
        <Status loading={loading} error={error} />
      ) : (
        <ScrollView contentContainerStyle={{ padding: 12 }}>
          <View style={s.row}>
            <Stat label="Matches" value={summary.data.matches.toLocaleString()} />
            <Stat label="Teams" value={summary.data.teams} />
            <Stat label="Seasons" value={summary.data.seasons} />
          </View>

          <Text style={s.section}>Team wins</Text>
          <View style={s.card2}>
            {wins.data.map((w) => (
              <BarRow
                key={w.team}
                label={w.team}
                value={w.wins}
                max={Math.max(...wins.data.map((x) => x.wins))}
              />
            ))}
          </View>

          <Text style={s.section}>Matches per season</Text>
          <View style={s.card2}>
            <ScrollView horizontal showsHorizontalScrollIndicator={false}>
              <View style={{ flexDirection: 'row', alignItems: 'flex-end', paddingTop: 8 }}>
                {seasons.data.map((d) => {
                  const maxM = Math.max(...seasons.data.map((x) => x.matches));
                  return (
                    <View key={d.year} style={{ alignItems: 'center', marginRight: 8, width: 30 }}>
                      <Text style={s.tiny}>{d.matches}</Text>
                      <View
                        style={{
                          width: 18,
                          height: Math.max((d.matches / maxM) * 110, 3),
                          backgroundColor: C.accent,
                          borderRadius: 3,
                        }}
                      />
                      <Text style={s.tiny}>{String(d.year).slice(2)}</Text>
                    </View>
                  );
                })}
              </View>
            </ScrollView>
          </View>
        </ScrollView>
      )}
    </View>
  );
}

// ---------- Matchups ----------
function MatchupCard({ item, showBatter }) {
  return (
    <View style={[s.card, { flexDirection: 'column', alignItems: 'flex-start' }]}>
      <Text style={s.name}>
        {showBatter ? `${item.batsman} vs ${item.bowler}` : item.bowler}
      </Text>
      <Text style={s.muted}>
        {item.balls_faced} balls · {item.runs_scored} runs · {item.dismissals} out
      </Text>
      <Text style={s.muted}>
        SR {item.strike_rate ?? '-'} · Avg {item.average ?? '-'}
      </Text>
    </View>
  );
}

function BatterPicker({ onPick }) {
  const [text, setText] = useState('');
  const q = useDebounced(text.trim());
  const { data, loading, error } = useApi(
    '/matchups/batters?limit=30' + (q ? '&search=' + encodeURIComponent(q) : '')
  );
  return (
    <View style={{ flex: 1 }}>
      <TextInput
        style={s.search}
        placeholder="Search a batter (e.g. Kohli)"
        placeholderTextColor={C.muted}
        value={text}
        onChangeText={setText}
      />
      {loading || error ? (
        <Status loading={loading} error={error} />
      ) : data.length === 0 ? (
        <View style={s.center}><Text style={s.muted}>No batters found.</Text></View>
      ) : (
        <FlatList
          data={data}
          keyExtractor={(item) => item.batsman}
          contentContainerStyle={{ padding: 12 }}
          renderItem={({ item }) => (
            <TouchableOpacity style={s.card} onPress={() => onPick(item.batsman)}>
              <Text style={[s.name, { flex: 1 }]}>{item.batsman}</Text>
              <Text style={s.muted}>{item.balls_faced} balls</Text>
            </TouchableOpacity>
          )}
        />
      )}
    </View>
  );
}

function BatterDetail({ name, onBack }) {
  const [text, setText] = useState('');
  const q = useDebounced(text.trim());
  const { data, loading, error } = useApi(
    '/matchups/by-batter?name=' + encodeURIComponent(name) +
    '&min_balls=6&limit=40' + (q ? '&bowler=' + encodeURIComponent(q) : '')
  );
  return (
    <View style={{ flex: 1 }}>
      <TouchableOpacity onPress={onBack} style={{ padding: 12, paddingBottom: 0 }}>
        <Text style={s.accentText}>← Back to batters</Text>
      </TouchableOpacity>
      <Text style={[s.section, { paddingHorizontal: 12 }]}>{name}</Text>
      <TextInput
        style={[s.search, { marginTop: 0 }]}
        placeholder="Filter by bowler (optional)"
        placeholderTextColor={C.muted}
        value={text}
        onChangeText={setText}
      />
      {loading || error ? (
        <Status loading={loading} error={error} />
      ) : (
        <FlatList
          data={data}
          keyExtractor={(item) => item.bowler}
          contentContainerStyle={{ padding: 12 }}
          renderItem={({ item }) => <MatchupCard item={item} />}
        />
      )}
    </View>
  );
}

function MostFaced() {
  const { data, loading, error } = useApi('/matchups/most-faced?limit=30');
  if (loading || error) return <Status loading={loading} error={error} />;
  return (
    <FlatList
      data={data}
      keyExtractor={(item) => item.batsman + item.bowler}
      contentContainerStyle={{ padding: 12 }}
      renderItem={({ item }) => <MatchupCard item={item} showBatter />}
    />
  );
}

function Matchups() {
  const [mode, setMode] = useState('Search a batter');
  const [picked, setPicked] = useState(null);
  return (
    <View style={{ flex: 1 }}>
      <Chips
        options={['Search a batter', 'Most faced']}
        value={mode}
        onChange={(m) => { setMode(m); setPicked(null); }}
      />
      {mode === 'Most faced' ? (
        <MostFaced />
      ) : picked ? (
        <BatterDetail name={picked} onBack={() => setPicked(null)} />
      ) : (
        <BatterPicker onPick={setPicked} />
      )}
    </View>
  );
}

// ---------- Archetypes ----------
function Archetypes() {
  const { data, loading, error } = useApi('/archetypes');
  if (loading || error) return <Status loading={loading} error={error} />;
  return (
    <ScrollView contentContainerStyle={{ padding: 12 }}>
      {data.map((c) => (
        <View key={c.cluster} style={[s.card, { flexDirection: 'column', alignItems: 'flex-start' }]}>
          <Text style={s.name}>Archetype {c.cluster}</Text>
          <Text style={s.accentText}>{c.players} batters · avg {Math.round(c.total_runs)} runs</Text>
          <Text style={s.muted}>Strike rate {c.strike_rate} · Boundaries {c.boundary_pct}%</Text>
          <Text style={s.muted}>
            Powerplay SR {c.powerplay_sr} · Middle SR {c.middle_sr} · Death SR {c.death_sr}
          </Text>
          <Text style={[s.text, { marginTop: 6 }]}>Top: {c.top_players.join(', ')}</Text>
        </View>
      ))}
    </ScrollView>
  );
}

// ---------- Venues ----------
function Venues() {
  const [text, setText] = useState('');
  const query = useDebounced(text.trim());
  const path = '/venues?limit=30' + (query ? '&search=' + encodeURIComponent(query) : '');
  const { data, loading, error } = useApi(path);

  return (
    <View style={{ flex: 1 }}>
      <TextInput
        style={s.search}
        placeholder="Search venue (e.g. Chennai)"
        placeholderTextColor={C.muted}
        value={text}
        onChangeText={setText}
      />
      {loading || error ? (
        <Status loading={loading} error={error} />
      ) : (
        <FlatList
          data={data}
          keyExtractor={(item) => item.venue}
          contentContainerStyle={{ padding: 12 }}
          renderItem={({ item }) => (
            <View style={[s.card, { flexDirection: 'column', alignItems: 'flex-start' }]}>
              <Text style={s.name}>{item.venue}</Text>
              <Text style={s.accentText}>{item.venue_character}</Text>
              <Text style={s.muted}>
                {item.total_matches} matches · avg 1st innings {item.avg_first_innings_score}
              </Text>
              <Text style={s.muted}>
                Chasing wins {item.chase_win_rate_pct}% · highest {item.highest_innings_total}
              </Text>
            </View>
          )}
        />
      )}
    </View>
  );
}

// ---------- Win predictor ----------
function oversToBalls(text) {
  const [o, b] = text.split('.');
  const overs = parseInt(o, 10);
  const balls = parseInt(b || '0', 10);
  if (isNaN(overs) || isNaN(balls) || balls > 5 || overs > 20) return null;
  return overs * 6 + balls;
}

function Predictor() {
  const [inning, setInning] = useState(2);
  const [score, setScore] = useState('');
  const [wickets, setWickets] = useState('');
  const [overs, setOvers] = useState('');
  const [target, setTarget] = useState('');
  const [result, setResult] = useState(null);
  const [err, setErr] = useState(null);
  const [busy, setBusy] = useState(false);

  async function predict() {
    setErr(null);
    setResult(null);
    const balls = oversToBalls(overs);
    const sc = parseInt(score, 10);
    const wk = parseInt(wickets, 10);
    const tg = parseInt(target, 10);
    if (balls === null || isNaN(sc) || isNaN(wk) || wk < 0 || wk > 10) {
      setErr('Enter valid numbers. Overs look like 10.3 (10 overs and 3 balls).');
      return;
    }
    if (inning === 2 && isNaN(tg)) {
      setErr('Enter the target for the 2nd innings.');
      return;
    }
    const body = {
      inning,
      current_score: sc,
      wickets_fallen: wk,
      balls_bowled: balls,
      ...(inning === 2 ? { target: tg } : {}),
    };
    setBusy(true);
    try {
      const r = await api('/predict/win-probability', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      setResult(r);
    } catch (e) {
      setErr(friendly(e));
    } finally {
      setBusy(false);
    }
  }

  const bat = result ? Math.round(result.batting_team_win_probability * 100) : 0;

  return (
    <ScrollView contentContainerStyle={{ padding: 16 }} keyboardShouldPersistTaps="handled">
      <Text style={s.label}>Innings</Text>
      <View style={s.row}>
        {[1, 2].map((n) => (
          <TouchableOpacity
            key={n}
            style={[s.toggle, inning === n && s.toggleOn]}
            onPress={() => setInning(n)}
          >
            <Text style={[s.toggleText, inning === n && { color: '#06210f' }]}>
              {n === 1 ? '1st innings' : '2nd innings (chase)'}
            </Text>
          </TouchableOpacity>
        ))}
      </View>

      <Text style={s.label}>Current score</Text>
      <TextInput style={s.input} keyboardType="number-pad" value={score} onChangeText={setScore}
        placeholder="e.g. 80" placeholderTextColor={C.muted} />

      <Text style={s.label}>Wickets fallen</Text>
      <TextInput style={s.input} keyboardType="number-pad" value={wickets} onChangeText={setWickets}
        placeholder="e.g. 3" placeholderTextColor={C.muted} />

      <Text style={s.label}>Overs bowled (e.g. 10.3)</Text>
      <TextInput style={s.input} keyboardType="decimal-pad" value={overs} onChangeText={setOvers}
        placeholder="e.g. 10.0" placeholderTextColor={C.muted} />

      {inning === 2 && (
        <>
          <Text style={s.label}>Target</Text>
          <TextInput style={s.input} keyboardType="number-pad" value={target} onChangeText={setTarget}
            placeholder="e.g. 170" placeholderTextColor={C.muted} />
        </>
      )}

      <TouchableOpacity style={s.button} onPress={predict} disabled={busy}>
        {busy ? <ActivityIndicator color="#06210f" /> : <Text style={s.buttonText}>Predict</Text>}
      </TouchableOpacity>

      {err && <Text style={[s.error, { marginTop: 14 }]}>{err}</Text>}

      {result && (
        <View style={[s.card, { flexDirection: 'column', alignItems: 'stretch', marginTop: 18 }]}>
          <View style={s.bar}>
            <View style={{ flex: bat, backgroundColor: C.accent }} />
            <View style={{ flex: 100 - bat, backgroundColor: C.bad }} />
          </View>
          <View style={[s.row, { justifyContent: 'space-between', marginTop: 10 }]}>
            <Text style={s.accentText}>Batting team {bat}%</Text>
            <Text style={{ color: C.bad, fontWeight: '700' }}>Bowling team {100 - bat}%</Text>
          </View>
        </View>
      )}
    </ScrollView>
  );
}

// ---------- App shell ----------
const TABS = [
  { key: 'over', label: 'Overview' },
  { key: 'bat', label: 'Batters' },
  { key: 'bowl', label: 'Bowlers' },
  { key: 'match', label: 'Matchups' },
  { key: 'arch', label: 'Archetypes' },
  { key: 'venue', label: 'Venues' },
  { key: 'live', label: 'Predict' },
];

export default function App() {
  const [tab, setTab] = useState('over');

  return (
    <View style={s.app}>
      <StatusBar barStyle="light-content" />
      <Text style={s.title}>🏏 Cricket Analytics</Text>

      <View style={{ flex: 1 }}>
        {tab === 'over' && <Overview />}
        {tab === 'bat' && (
          <PlayerList
            path="/players/top-batters?limit=25&min_runs=200"
            nameKey="batsman"
            impactKey="batting_impact"
            line={(p) => `${p.total_runs} runs · SR ${p.strike_rate} · avg ${p.avg_runs_per_innings}/inn`}
          />
        )}
        {tab === 'bowl' && (
          <PlayerList
            path="/players/top-bowlers?limit=25&min_wickets=10"
            nameKey="bowler"
            impactKey="bowling_impact"
            line={(p) => `${p.wickets} wkts · economy ${p.economy} · ${p.wickets_per_innings}/inn`}
          />
        )}
        {tab === 'match' && <Matchups />}
        {tab === 'arch' && <Archetypes />}
        {tab === 'venue' && <Venues />}
        {tab === 'live' && <Predictor />}
      </View>

      <View style={s.tabBar}>
        <ScrollView horizontal showsHorizontalScrollIndicator={false}>
          {TABS.map((t) => (
            <TouchableOpacity key={t.key} style={s.tab} onPress={() => setTab(t.key)}>
              <Text style={[s.tabText, tab === t.key && { color: C.accent }]}>{t.label}</Text>
            </TouchableOpacity>
          ))}
        </ScrollView>
      </View>
    </View>
  );
}

// ---------- Styles ----------
const s = StyleSheet.create({
  app: { flex: 1, backgroundColor: C.bg, paddingTop: Platform.OS === 'android' ? StatusBar.currentHeight : 50 },
  title: { color: C.text, fontSize: 22, fontWeight: '800', paddingHorizontal: 16, paddingBottom: 8 },
  center: { flex: 1, alignItems: 'center', justifyContent: 'center', padding: 24 },
  text: { color: C.text, fontSize: 14 },
  muted: { color: C.muted, fontSize: 13 },
  tiny: { color: C.muted, fontSize: 10, marginVertical: 2 },
  error: { color: C.bad, fontSize: 15, textAlign: 'center' },
  accentText: { color: C.accent, fontWeight: '700', marginVertical: 2 },
  section: { color: C.text, fontSize: 18, fontWeight: '800', marginTop: 18, marginBottom: 8 },
  card: { backgroundColor: C.card, borderRadius: 12, padding: 14, marginBottom: 10, flexDirection: 'row', alignItems: 'center' },
  card2: { backgroundColor: C.card, borderRadius: 12, padding: 14 },
  statCard: { flex: 1, backgroundColor: C.card, borderRadius: 12, padding: 14, marginRight: 8, alignItems: 'center' },
  statValue: { color: C.accent, fontSize: 22, fontWeight: '800' },
  rank: { color: C.muted, width: 28, fontSize: 16, fontWeight: '700' },
  name: { color: C.text, fontSize: 16, fontWeight: '700' },
  impact: { color: C.accent, fontSize: 22, fontWeight: '800', marginLeft: 8 },
  search: { backgroundColor: C.card, color: C.text, margin: 12, marginBottom: 0, padding: 12, borderRadius: 10, fontSize: 16 },
  chip: { paddingVertical: 8, paddingHorizontal: 16, borderRadius: 20, backgroundColor: C.card, marginRight: 8 },
  chipOn: { backgroundColor: C.accent },
  chipText: { color: C.text, fontWeight: '700' },
  track: { height: 8, backgroundColor: C.bg, borderRadius: 4, marginTop: 4, overflow: 'hidden' },
  fill: { height: 8, backgroundColor: C.accent, borderRadius: 4 },
  label: { color: C.muted, marginTop: 14, marginBottom: 6, fontSize: 13 },
  input: { backgroundColor: C.card, color: C.text, padding: 12, borderRadius: 10, fontSize: 16 },
  row: { flexDirection: 'row' },
  toggle: { flex: 1, padding: 12, borderRadius: 10, backgroundColor: C.card, marginRight: 8, alignItems: 'center' },
  toggleOn: { backgroundColor: C.accent },
  toggleText: { color: C.text, fontWeight: '600' },
  button: { backgroundColor: C.accent, padding: 15, borderRadius: 12, alignItems: 'center', marginTop: 22 },
  buttonText: { color: '#06210f', fontSize: 17, fontWeight: '800' },
  bar: { flexDirection: 'row', height: 18, borderRadius: 9, overflow: 'hidden' },
  tabBar: { backgroundColor: C.card, paddingBottom: Platform.OS === 'ios' ? 20 : 6 },
  tab: { paddingVertical: 14, paddingHorizontal: 16, alignItems: 'center' },
  tabText: { color: C.muted, fontWeight: '700' },
});

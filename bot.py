# ==============================================================================
# 🤖 PROJECT: THE ULTIMATE GUARDIAN (FINAL SYNTAX FIX)
# ==============================================================================

import os
import json
import random
import discord
from discord import app_commands
from discord.ext import commands, tasks
import asyncio
from flask import Flask
from threading import Thread
import google.generativeai as genai
import pandas as pd
from datetime import datetime
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM
from duckduckgo_search import DDGS

# --- 🌐 WEB SERVER ---
app = Flask('')
@app.route('/')
def home(): return "SYSTEM ONLINE: Ultimate Engine Active."
def run(): app.run(host='0.0.0.0', port=8080)
def keep_alive(): t = Thread(target=run); t.start()

# --- 🔑 CONFIGURATION ---
DISCORD_TOKEN = os.environ.get('DISCORD_TOKEN')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
PORTFOLIOS_FILE = 'user_portfolios.json'
TRADING_FILE = 'paper_trading.json'
START_DATE = '2020-01-01'
PREDICTION_DAYS = 60

try:
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-pro')
except: pass

class MyBot(commands.Bot):
    def __init__(self):
        super().__init__(command_prefix='!', intents=discord.Intents.default())
    async def setup_hook(self):
        await self.tree.sync() # บรรทัดนี้สำคัญ! มันจะซิงค์คำสั่ง / ไปที่ Discord

bot = MyBot()

# ==============================================================================
# 🦆 FREE SEARCH ENGINE (DUCKDUCKGO)
# ==============================================================================
def search_free_intel(ticker):
    news_text = ""
    try:
        keywords = f"{ticker} stock news rumors analysis"
        results = DDGS().text(keywords, max_results=3)
        if results:
            for r in results:
                news_text += f"- [Web] {r.get('title','')}: {r.get('body','')}\n"
    except Exception as e:
        print(f"DDG Error: {e}")
    return news_text

# ==============================================================================
# 🗄️ CORE LOGIC
# ==============================================================================
def get_usd_thb_rate():
    try: return yf.Ticker("THB=X").history(period="1d")['Close'].iloc[-1]
    except: return 34.5

def load_json(f): return json.load(open(f)) if os.path.exists(f) else {}
def save_json(f, d): json.dump(d, open(f,'w'), indent=4)

def get_trader_profile(user_id):
    d = load_json(TRADING_FILE)
    if str(user_id) not in d:
        d[str(user_id)] = {"balance": 100000.0, "portfolio": {}, "history": []}
        save_json(TRADING_FILE, d)
    return d[str(user_id)]

def check_fundamentals(ticker):
    try:
        info = yf.Ticker(ticker).info
        score = 0; status = "พื้นฐานแกร่ง 💪"
        m = info.get('profitMargins', 0); g = info.get('revenueGrowth', 0); pe = info.get('trailingPE', 0)
        if m > 0: score += 1
        if g > 0: score += 1
        if 0 < pe < 60: score += 1
        if m < 0: status = "ขาดทุน 🩸"; score = -1
        elif score == 0: status = "น่าห่วง 😷"
        return {"status": status, "score": score}
    except: return {"status": "ไม่ทราบ", "score": 0}

def analyze_chain_reaction(ticker):
    if not GEMINI_API_KEY: return "AI Disconnected"
    prompt = f"Analyze 'Chain Reaction' for stock {ticker} in Thai. 1. Suppliers/Customers? 2. If {ticker} crashes, who falls? Short & Concise."
    try: return model.generate_content(prompt).text
    except: return "AI Error"

def analyze_market(ticker):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    try:
        df = yf.download(ticker, start=START_DATE, progress=False)
        if len(df) < 100: return None
        curr_price = df['Close'].iloc[-1].item()
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi = (100 - (100 / (1 + (gain / loss)))).iloc[-1].item()
        
        data = df[['Close']].values
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(data)
        X = np.array([scaled[-PREDICTION_DAYS:]])
        
        m_file = f'brain_{ticker}.keras'
        if not os.path.exists(m_file):
            ai = Sequential([LSTM(50, input_shape=(PREDICTION_DAYS, 1)), Dense(1)])
            ai.compile(optimizer='adam', loss='mse')
            ai.fit(X, np.array([scaled[-1]]), epochs=5, verbose=0)
            ai.save(m_file)
        else: ai = load_model(m_file)
        pred = scaler.inverse_transform(ai.predict(X, verbose=0))[0][0]
        
        fund = check_fundamentals(ticker)
        stock = yf.Ticker(ticker)
        insider = stock.insider_transactions
        insider_txt = "ปกติ"
        if insider is not None and not insider.empty:
             if sum(1 for i, r in insider.head(5).iterrows() if "sale" in str(r.get('Transaction','')).lower()) >= 2:
                 insider_txt = "🚨 เทขาย"

        news_report = {"summary": "ไม่มีข่าว", "impact": "Low", "panic": False, "advice": "ถือต่อ"}
        all_headlines = ""
        try:
            news = stock.news[:3]
            if news: all_headlines += "\n".join([f"- [Yahoo] {n['title']}" for n in news])
        except: pass

        all_headlines += "\n" + search_free_intel(ticker)

        if GEMINI_API_KEY and all_headlines.strip():
            try:
                prompt = f"Analyze mixed news for '{ticker}':\n{all_headlines}\nTask: Summarize (Thai), Impact (Low/Med/Crit), Panic(YES/NO), Advice. Format:\nSummary: ...\nImpact: ...\nPanic: ...\nAdvice: ..."
                res = model.generate_content(prompt).text
                lines = res.split('\n')
                for line in lines:
                    if "Summary:" in line: news_report['summary'] = line.replace("Summary:", "").strip()
                    if "Impact:" in line: news_report['impact'] = line.replace("Impact:", "").strip()
                    if "Panic:" in line: news_report['panic'] = "YES" in line.upper()
                    if "Advice:" in line: news_report['advice'] = line.replace("Advice:", "").strip()
            except: pass

        ai_score = np.clip(((pred - curr_price)/curr_price)*10, -1, 1)
        rsi_score = 1 if rsi < 30 else -1 if rsi > 70 else 0
        fund_adj = (fund['score']-1)/2
        panic_score = -1.0 if news_report['panic'] else 0.2
        
        final_score = (ai_score*0.3) + (rsi_score*0.2) + (fund_adj*0.3) + (panic_score*0.2)
        if fund['score'] == -1: final_score = -1.0
        
        signal = "HOLD ✋"; color = 0x95a5a6
        if final_score > 0.35: signal = "BUY 🟢"; color = 0x2ecc71
        elif final_score < -0.35: signal = "SELL 🔴"; color = 0xe74c3c
        if news_report['panic']: signal = "PANIC SELL 🚨"; color = 0xff0000

        return {
            "price": curr_price, "signal": signal, "score": final_score, 
            "color": color, "fund": fund, "insider": insider_txt,
            "news_report": news_report, "news_raw": all_headlines
        }
    except: return None

# ==============================================================================
# 🎮 MODERN UI
# ==============================================================================
class TradeModal(discord.ui.Modal):
    def __init__(self, ticker, action):
        super().__init__(title=f"{action} Stock")
        self.ticker = ticker; self.action = action
        self.amount = discord.ui.TextInput(label="Amount", placeholder="10", required=True)
        self.price = discord.ui.TextInput(label="Price", placeholder="Optional", required=False)
        self.add_item(self.amount); self.add_item(self.price)
    async def on_submit(self, interaction: discord.Interaction):
        try:
            amt = int(self.amount.value)
            p = float(self.price.value) if self.price.value else None
            data = load_json(TRADING_FILE); prof = data.get(str(interaction.user.id))
            if not prof: await interaction.response.send_message("❌ Register first", ephemeral=True); return
            rate = get_usd_thb_rate()
            if self.action == "BUY":
                curr = p if p else yf.Ticker(self.ticker).history(period="1d")['Close'].iloc[-1]
                cost = curr*amt
                if not p and prof['balance'] < cost: await interaction.response.send_message("❌ No Money", ephemeral=True); return
                prof['balance'] -= cost
                if self.ticker in prof['portfolio']:
                    old = prof['portfolio'][self.ticker]; oa = old['amount'] if isinstance(old, dict) else old; oc = old['cost'] if isinstance(old, dict) else curr
                    prof['portfolio'][self.ticker] = {"amount": oa+amt, "cost": ((oa*oc)+(amt*curr))/(oa+amt)}
                else: prof['portfolio'][self.ticker] = {"amount": amt, "cost": curr}
                save_json(TRADING_FILE, data)
                await interaction.response.send_message(f"✅ BOUGHT {amt} {self.ticker}")
            elif self.action == "SELL":
                if self.ticker not in prof['portfolio']: await interaction.response.send_message("❌ No Stock", ephemeral=True); return
                item = prof['portfolio'][self.ticker]; oa = item['amount'] if isinstance(item, dict) else item
                if oa < amt: await interaction.response.send_message("❌ Not enough", ephemeral=True); return
                curr = p if p else yf.Ticker(self.ticker).history(period="1d")['Close'].iloc[-1]
                income = curr*amt; prof['balance'] += income
                if oa == amt: del prof['portfolio'][self.ticker]
                else: prof['portfolio'][self.ticker]['amount'] = oa - amt
                save_json(TRADING_FILE, data)
                await interaction.response.send_message(f"✅ SOLD {amt} {self.ticker}")
        except: await interaction.response.send_message("❌ Error", ephemeral=True)

class TradeButtons(discord.ui.View):
    def __init__(self, ticker): super().__init__(timeout=None); self.ticker = ticker
    @discord.ui.button(label="Buy", style=discord.ButtonStyle.green, emoji="💵")
    async def b(self, i: discord.Interaction, b: discord.ui.Button): await i.response.send_modal(TradeModal(self.ticker, "BUY"))
    @discord.ui.button(label="Sell", style=discord.ButtonStyle.red, emoji="📉")
    async def s(self, i: discord.Interaction, b: discord.ui.Button): await i.response.send_modal(TradeModal(self.ticker, "SELL"))
    @discord.ui.button(label="Watch", style=discord.ButtonStyle.blurple, emoji="⭐")
    async def w(self, i: discord.Interaction, b: discord.ui.Button):
        uid=str(i.user.id); p=load_json(PORTFOLIOS_FILE); up=p.get(uid,[])
        if self.ticker not in up: up.append(self.ticker); p[uid]=up; save_json(PORTFOLIOS_FILE,p); await i.response.send_message("✅ Added", ephemeral=True)
        else: await i.response.send_message("👀 Watching", ephemeral=True)

# ==============================================================================
# 📱 COMMANDS
# ==============================================================================
@bot.tree.command(name="check", description="Full Analysis (Intel + Chain Reaction)")
async def check(interaction: discord.Interaction, ticker: str):
    await interaction.response.defer()
    t = ticker.upper()
    loop = asyncio.get_running_loop()
    d = await loop.run_in_executor(None, analyze_market, t)
    chain_res = await loop.run_in_executor(None, analyze_chain_reaction, t)
    
    if d:
        thb = d['price'] * get_usd_thb_rate()
        rpt = d['news_report']
        emb_color = 0xff0000 if rpt['panic'] else d['color']
        
        em = discord.Embed(title=f"💎 Report: {t}", color=emb_color)
        em.add_field(name="📊 Data", value=f"Price: ${d['price']:.2f} (฿{thb:,.0f})\nSignal: **{d['signal']}**", inline=True)
        
        panic_icon = "😱 PANIC!" if rpt['panic'] else "😌 CHILL"
        em.add_field(name="🕵️ Intel Report", value=f"Summary: {rpt['summary']}\nPanic: {panic_icon}\nAdvice: {rpt['advice']}", inline=False)
        em.add_field(name="🔗 Chain Reaction", value=chain_res, inline=False)
        if d['news_raw']: em.add_field(name="🌐 Sources", value=d['news_raw'][:300] + "...", inline=False)

        await interaction.followup.send(embed=em, view=TradeButtons(t))
    else: await interaction.followup.send("❌ Error")

@bot.tree.command(name="wallet", description="View Portfolio")
async def wallet(interaction: discord.Interaction):
    await interaction.response.defer()
    d = get_trader_profile(interaction.user.id); rate = get_usd_thb_rate()
    embed = discord.Embed(title=f"🇹🇭 Portfolio", color=0x3498db)
    curr_val = 0; txt = ""
    if d['portfolio']:
        for t, info in d['portfolio'].items():
            try: cp = yf.Ticker(t).history(period="1d")['Close'].iloc[-1]
            except: cp = 0
            if isinstance(info, dict): amt=info['amount']; cost=info['cost']
            else: amt=info; cost=cp
            prof = (amt*cp)-(amt*cost); icon = "🟢" if prof >= 0 else "🔴"
            txt += f"**{t}**: {amt} | ${cost:.2f} | {icon} ${prof:+,.2f}\n"
            curr_val += amt*cp
    cash = d['balance']
    embed.add_field(name=f"💵 Cash", value=f"${cash:,.2f} (฿{cash*rate:,.0f})", inline=False)
    if txt: embed.add_field(name="📈 Stocks", value=txt, inline=False)
    embed.set_footer(text=f"Net Worth: ${cash+curr_val:,.2f}")
    await interaction.followup.send(embed=embed)

@bot.tree.command(name="register", description="Open Account")
async def register(interaction: discord.Interaction):
    d = get_trader_profile(interaction.user.id)
    await interaction.response.send_message(f"🎉 Ready! Balance: ${d['balance']:,.2f}")

@bot.tree.command(name="setcost", description="Fix cost")
async def setcost(interaction: discord.Interaction, ticker: str, price: float):
    d = load_json(TRADING_FILE); uid = str(interaction.user.id); t = ticker.upper()
    if uid in d and t in d[uid]['portfolio']:
        d[uid]['portfolio'][t]['cost'] = price; save_json(TRADING_FILE, d)
        await interaction.response.send_message(f"✅ Updated")
    else: await interaction.response.send_message("❌ Not found")

@bot.tree.command(name="scan", description="Scan Watchlist")
async def scan(interaction: discord.Interaction):
    await interaction.response.defer()
    p = load_json(PORTFOLIOS_FILE).get(str(interaction.user.id), [])
    if not p: await interaction.followup.send("📭 Empty"); return
    await interaction.followup.send(f"🔎 Scanning {len(p)} stocks...")
    loop = asyncio.get_running_loop()
    for t in p:
        d = await loop.run_in_executor(None, analyze_market, t)
        if d:
            rpt = d['news_report']
            icon = "🚨" if rpt['panic'] else "✅"
            em = discord.Embed(title=f"{t}: {d['signal']}", color=d['color'])
            em.description=f"Price: ${d['price']:.2f}\n{icon} Panic: {rpt['panic']}"
            await interaction.followup.send(embed=em)

@bot.tree.command(name="teach", description="Teach terms")
async def teach(interaction: discord.Interaction, term: str):
    await interaction.response.defer()
    if GEMINI_API_KEY: await interaction.followup.send(f"🎓 **{term}:**\n{model.generate_content(f'Teach {term} in Thai simple').text}")
    else: await interaction.followup.send("No API")

# ==============================================================================
# 🔔 ALERTS (FIXED INDENTATION)
# ==============================================================================
@tasks.loop(seconds=1) 
async def ninja_alert_task():
    await bot.wait_until_ready()
    while not bot.is_closed():
        portfolios = load_json(PORTFOLIOS_FILE)
        all_t = set()
        
        # --- จุดที่แก้: เขียนแยกบรรทัดให้ถูกต้อง ---
        for t_list in portfolios.values(): 
            for t in t_list: 
                all_t.add(t)
        
        if not all_t: await asyncio.sleep(60); continue
        
        for t in all_t:
            try:
                await asyncio.sleep(random.uniform(5, 10))
                loop = asyncio.get_running_loop()
                d = await loop.run_in_executor(None, analyze_market, t)
                
                if d and ("NOW" in d['signal'] or d['news_report']['panic']):
                    rpt = d['news_report']
                    is_panic = rpt['panic']
                    title = f"🚨 PANIC: {t}" if is_panic else f"📢 ALERT: {t}"
                    color = 0xff0000 if is_panic else d['color']
                    
                    em = discord.Embed(title=title, color=color)
                    em.add_field(name="สถานการณ์", value=rpt['summary'], inline=False)
                    em.add_field(name="คำแนะนำ", value=f"👉 **{rpt['advice']}**", inline=False)
                    em.set_footer(text="Ultimate Watchdog")
                    
                    for uid, ts in portfolios.items():
                        if t in ts:
                            try: await bot.get_user(int(uid)).send(embed=em)
                            except: pass
            except: pass
        await asyncio.sleep(random.uniform(30, 60))

@bot.event
async def on_ready():
    print(f'🤖 Ultimate Bot Online: {bot.user}')
    if not ninja_alert_task.is_running(): ninja_alert_task.start()

if __name__ == "__main__":
    keep_alive()
    if DISCORD_TOKEN: bot.run(DISCORD_TOKEN)

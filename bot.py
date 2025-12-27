# ==============================================================================
# 🤖 PROJECT: SMART INVESTOR GUARDIAN (ULTIMATE FUSION: JUDGE + LEARNING + PORTFOLIO)
# ==============================================================================

import os
import json
import discord
from discord.ext import commands
import asyncio
from flask import Flask
from threading import Thread
import google.generativeai as genai
import pandas as pd
from datetime import datetime
import numpy as np

# --- 🌐 WEB SERVER ---
app = Flask('')
@app.route('/')
def home(): return "SYSTEM ONLINE: The Ultimate Guardian is watching."
def run(): app.run(host='0.0.0.0', port=8080)
def keep_alive(): t = Thread(target=run); t.start()

# --- 🔑 CONFIGURATION ---
DISCORD_TOKEN = os.environ.get('DISCORD_TOKEN')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
PORTFOLIOS_FILE = 'user_portfolios.json'
HISTORY_FILE = 'prediction_history.json'
START_DATE = '2020-01-01'
PREDICTION_DAYS = 60

# ตั้งค่า Gemini
try:
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-pro')
    else: print("⚠️ Warning: GEMINI_API_KEY not found")
except: pass

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

# ==============================================================================
# 💾 DATABASE SYSTEM (PORTFOLIO & LEARNING)
# ==============================================================================
def load_json(filename):
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f: return json.load(f)
        except: return {}
    return {}

def save_json(filename, data):
    with open(filename, 'w') as f: json.dump(data, f)

def log_prediction(ticker, signal, price):
    """จดจำการทำนายเพื่อเรียนรู้"""
    history = load_json(HISTORY_FILE)
    if ticker not in history: history[ticker] = []
    
    # บันทึกเฉพาะสัญญาณซื้อ/ขายที่ชัดเจน
    if "BUY" in signal or "SELL" in signal:
        record = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "signal": signal,
            "entry_price": price,
            "status": "PENDING"
        }
        history[ticker].append(record)
        save_json(HISTORY_FILE, history)

def get_accuracy_stats(ticker, current_price):
    """ตรวจการบ้าน: คำนวณ Win Rate"""
    history = load_json(HISTORY_FILE)
    records = history.get(ticker, [])
    if not records: return "New Stock (No History)"
    
    # อัปเดตสถานะ (ตรวจคำตอบ)
    changed = False
    correct = 0
    total_checked = 0
    
    for r in records:
        if r['status'] == 'PENDING':
            # กฎ: กำไร 1% ถือว่าถูก, ขาดทุนถือว่าผิด
            if "BUY" in r['signal']:
                if current_price > r['entry_price'] * 1.01: r['status'] = 'CORRECT'; changed=True
                elif current_price < r['entry_price']: r['status'] = 'WRONG'; changed=True
            elif "SELL" in r['signal']:
                if current_price < r['entry_price'] * 0.99: r['status'] = 'CORRECT'; changed=True
                elif current_price > r['entry_price']: r['status'] = 'WRONG'; changed=True
        
        if r['status'] == 'CORRECT': correct += 1
        if r['status'] != 'PENDING': total_checked += 1
            
    if changed: save_json(HISTORY_FILE, history)
    
    if total_checked == 0: return "Waiting for results..."
    win_rate = (correct / total_checked) * 100
    return f"🏆 Win Rate: {win_rate:.1f}% ({correct}/{total_checked})"

# ==============================================================================
# 🧠 CORE ENGINE (LOCAL AI + INSIDER + NEWS)
# ==============================================================================
def get_insider_activity(ticker):
    import yfinance as yf
    try:
        stock = yf.Ticker(ticker)
        insider = stock.insider_transactions
        if insider is None or insider.empty: return "ไม่พบข้อมูล"
        latest = insider.head(3)
        summary = ""
        for i, r in latest.iterrows():
            summary += f"- {str(i)[:10]}: {r.get('Insider','?')} ({r.get('Transaction','?')}) {r.get('Shares',0)} หุ้น\n"
        return summary
    except: return "N/A"

def get_news_summary(ticker):
    import yfinance as yf
    try:
        stock = yf.Ticker(ticker)
        return "\n".join([f"- {n['title']}" for n in stock.news[:3]])
    except: return "N/A"

def analyze_technical(ticker):
    """วิเคราะห์กราฟด้วย LSTM และ RSI"""
    import yfinance as yf
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Dense, LSTM
    import os
    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    try:
        df = yf.download(ticker, start=START_DATE, progress=False)
        if len(df) < 100: return None
        
        curr_price = df['Close'].iloc[-1].item()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / loss)))
        curr_rsi = df['RSI'].iloc[-1].item()
        
        # LSTM Prediction
        data = df[['Close']].values
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)
        
        X = np.array([scaled_data[-PREDICTION_DAYS:]])
        
        model_file = f'brain_{ticker}.keras'
        if os.path.exists(model_file):
            ai_model = load_model(model_file)
        else:
            ai_model = Sequential([LSTM(50, input_shape=(PREDICTION_DAYS, 1)), Dense(1)])
            ai_model.compile(optimizer='adam', loss='mse')
            ai_model.fit(X, np.array([scaled_data[-1]]), epochs=1, verbose=0)
            ai_model.save(model_file)
            
        pred = ai_model.predict(X, verbose=0)
        pred_price = scaler.inverse_transform(pred)[0][0]
        
        # สร้าง Signal เบื้องต้น
        trend = "UP 🟢" if pred_price > curr_price else "DOWN 🔴"
        signal = "HOLD"
        if pred_price > curr_price * 1.01 and curr_rsi < 65: signal = "BUY NOW 🟢"
        elif pred_price < curr_price * 0.99 or curr_rsi > 75: signal = "SELL NOW 🔴"
        
        return {
            "price": curr_price, "ai_price": pred_price, 
            "rsi": curr_rsi, "trend": trend, "signal": signal
        }
    except Exception as e: return None

# ==============================================================================
# ⚖️ THE JUDGE (GEMINI)
# ==============================================================================
def consult_judge(ticker, tech, insider, news, stats):
    if not GEMINI_API_KEY: return "⚠️ (Gemini Disabled) เชื่อกราฟไปก่อนครับ"
    
    prompt = f"""
    วิเคราะห์หุ้น {ticker} สั้นๆ แบบเซียนหุ้น:
    1. กราฟ: ราคา ${tech['price']:.2f}, AIมอง: {tech['trend']}, RSI: {tech['rsi']:.1f} ({tech['signal']})
    2. สถิติความแม่นยำบอท: {stats}
    3. ผู้บริหาร: {insider}
    4. ข่าว: {news}
    
    ขอคำแนะนำ 3 ส่วน:
    - สถานการณ์: (สั้นๆ)
    - ความเสี่ยง: (สิ่งที่น่าห่วง)
    - คำตัดสิน: (ฟันธงว่า เชื่อกราฟดีไหม หรือควรระวังข่าว/ผู้บริหาร)
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except: return "Gemini กำลังพักผ่อน... (Error calling API)"

# ==============================================================================
# 🎮 DISCORD COMMANDS
# ==============================================================================
@bot.event
async def on_ready(): print(f'🤖 Ultimate Bot Online: {bot.user}')

@bot.command()
async def add(ctx, *tickers):
    """(Multi-User) เพิ่มหุ้นเข้าพอร์ตส่วนตัว"""
    user_id = str(ctx.author.id)
    port = load_json(PORTFOLIOS_FILE)
    user_port = port.get(user_id, [])
    
    added = []
    for t in tickers:
        t = t.upper().replace(",", "")
        if t not in user_port:
            user_port.append(t)
            added.append(t)
            
    port[user_id] = user_port
    save_json(PORTFOLIOS_FILE, port)
    if added: await ctx.send(f"✅ เพิ่ม {', '.join(added)} เข้าพอร์ตของคุณ {ctx.author.name} แล้ว")

@bot.command()
async def remove(ctx, *tickers):
    """(Multi-User) ลบหุ้นออกจากพอร์ต"""
    user_id = str(ctx.author.id)
    port = load_json(PORTFOLIOS_FILE)
    user_port = port.get(user_id, [])
    
    removed = []
    for t in tickers:
        t = t.upper().replace(",", "")
        if t in user_port:
            user_port.remove(t)
            removed.append(t)
            
    port[user_id] = user_port
    save_json(PORTFOLIOS_FILE, port)
    await ctx.send(f"🗑️ ลบ {', '.join(removed)} เรียบร้อย")

@bot.command()
async def port(ctx):
    """(Fast) สแกนพอร์ตส่วนตัวแบบเร็ว (ไม่ใช้ Gemini เพื่อประหยัดเวลา)"""
    user_id = str(ctx.author.id)
    port = load_json(PORTFOLIOS_FILE).get(user_id, [])
    
    if not port:
        await ctx.send("📭 พอร์ตว่างเปล่า (ใช้ !add เพื่อเพิ่มหุ้น)")
        return

    await ctx.send(f"🚀 กำลังสแกนพอร์ต {len(port)} ตัว ของคุณ {ctx.author.name}...")
    
    for ticker in port:
        loop = asyncio.get_running_loop()
        tech = await loop.run_in_executor(None, analyze_technical, ticker)
        
        if tech:
            stats = get_accuracy_stats(ticker, tech['price']) # เช็คความแม่นยำ
            color = 0x2ecc71 if "BUY" in tech['signal'] else 0xe74c3c if "SELL" in tech['signal'] else 0x95a5a6
            
            embed = discord.Embed(title=f"📊 {ticker}", color=color)
            embed.add_field(name="Price", value=f"${tech['price']:.2f}", inline=True)
            embed.add_field(name="Signal", value=f"**{tech['signal']}**", inline=True)
            embed.add_field(name="Bot Accuracy", value=stats, inline=True)
            await ctx.send(embed=embed)

@bot.command()
async def check(ctx, ticker: str):
    """(Deep Dive) วิเคราะห์เจาะลึกด้วย Gemini + Learning"""
    ticker = ticker.upper()
    msg = await ctx.send(f"⚖️ **กำลังเปิดศาลไต่สวน {ticker}...**\n(เรียกกราฟ.. สืบ Insider.. อ่านข่าว..)")
    
    loop = asyncio.get_running_loop()
    
    # 1. หาข้อมูลทั้งหมดพร้อมกัน
    tech_task = loop.run_in_executor(None, analyze_technical, ticker)
    insider_task = loop.run_in_executor(None, get_insider_activity, ticker)
    news_task = loop.run_in_executor(None, get_news_summary, ticker)
    
    tech, insider, news = await asyncio.gather(tech_task, insider_task, news_task)
    
    if not tech:
        await msg.edit(content=f"❌ ไม่พบข้อมูลหุ้น {ticker}")
        return

    # 2. คำนวณความแม่นยำ และ บันทึกผลการทาย (Self-Learning)
    stats = get_accuracy_stats(ticker, tech['price'])
    log_prediction(ticker, tech['signal'], tech['price']) # <--- จดบันทึกตรงนี้!

    # 3. ให้ Gemini ตัดสิน
    verdict = await loop.run_in_executor(None, consult_judge, ticker, tech, insider, news, stats)
    
    # 4. แสดงผล
    embed = discord.Embed(title=f"🏛️ คำพิพากษา: {ticker}", color=0xf1c40f)
    embed.add_field(name="💰 ราคา / RSI", value=f"${tech['price']:.2f} / {tech['rsi']:.1f}", inline=True)
    embed.add_field(name="🤖 บอททายว่า", value=f"**{tech['signal']}**", inline=True)
    embed.add_field(name="🏆 ความแม่นในอดีต", value=stats, inline=True)
    
    embed.description = f"**👨‍⚖️ มุมมองจาก The Judge:**\n{verdict}"
    embed.set_footer(text=f"Insider & News included • Self-Learning Active")
    
    await ctx.send(embed=embed)
    await msg.delete()

if __name__ == "__main__":
    keep_alive()
    if DISCORD_TOKEN:
        bot.run(DISCORD_TOKEN)

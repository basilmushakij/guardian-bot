# ==============================================================================
# 🤖 PROJECT: OMNISCIENT GUARDIAN (GENIUS MENTOR EDITION)
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
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM

# --- 🌐 WEB SERVER ---
app = Flask('')
@app.route('/')
def home(): return "SYSTEM ONLINE: Genius Mentor Active."
def run(): app.run(host='0.0.0.0', port=8080)
def keep_alive(): t = Thread(target=run); t.start()

# --- 🔑 CONFIGURATION ---
DISCORD_TOKEN = os.environ.get('DISCORD_TOKEN')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
PORTFOLIOS_FILE = 'user_portfolios.json'
HISTORY_FILE = 'prediction_history.json'
START_DATE = '2020-01-01'
PREDICTION_DAYS = 60

# ตั้งค่า Gemini (สมองหลัก)
try:
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-pro')
except: pass

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

# ==============================================================================
# 🗄️ INTELLIGENT MEMORY SYSTEM
# ==============================================================================
def load_json(filename):
    if os.path.exists(filename):
        try: with open(filename, 'r') as f: return json.load(f)
        except: return {}
    return {}

def save_json(filename, data):
    with open(filename, 'w') as f: json.dump(data, f)

def get_learning_stats(ticker):
    """ดึงสถิติความแม่นยำ เพื่อบอก user ว่าบอทน่าเชื่อแค่ไหน"""
    history = load_json(HISTORY_FILE)
    records = history.get(ticker, [])
    if not records: return "ยังไม่มีข้อมูล (หุ้นใหม่สำหรับบอท)"
    
    correct = 0
    total = 0
    current_price = yf.Ticker(ticker).history(period="1d")['Close'].iloc[-1]
    
    changed = False
    for r in records:
        if r['status'] == 'PENDING':
            # Logic การตรวจการบ้าน
            entry = r['entry_price']
            if "BUY" in r['signal']:
                if current_price > entry * 1.02: r['status']='CORRECT'; changed=True
                elif current_price < entry * 0.98: r['status']='WRONG'; changed=True
            elif "SELL" in r['signal']:
                if current_price < entry * 0.98: r['status']='CORRECT'; changed=True
                elif current_price > entry * 1.02: r['status']='WRONG'; changed=True
        
        if r['status'] == 'CORRECT': correct += 1
        if r['status'] != 'PENDING': total += 1
            
    if changed: save_json(HISTORY_FILE, history)
    
    if total == 0: return "รอผลการทดสอบ..."
    win_rate = (correct / total) * 100
    
    # แปลผล Win Rate เป็นภาษาคน
    if win_rate > 70: return f"แม่นยำสูง 🔥 ({win_rate:.0f}%)"
    elif win_rate > 50: return f"พอใช้ได้ 😐 ({win_rate:.0f}%)"
    else: return f"บอทยังเดาผิดบ่อย 🥶 ({win_rate:.0f}%)"

def log_signal(ticker, signal, price):
    if "HOLD" in signal: return
    history = load_json(HISTORY_FILE)
    if ticker not in history: history[ticker] = []
    history[ticker].append({
        "date": datetime.now().strftime("%Y-%m-%d"),
        "signal": signal,
        "entry_price": price,
        "status": "PENDING"
    })
    save_json(HISTORY_FILE, history)

# ==============================================================================
# 🕵️ ADVANCED DATA GATHERING
# ==============================================================================
def get_stock_profile(ticker):
    """ตรวจสอบว่าเป็นหุ้นปั่นหรือหุ้นดี (Safety Check)"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # 1. Market Cap (ขนาดบริษัท)
        mcap = info.get('marketCap', 0)
        is_small_cap = mcap < 2000000000 # ต่ำกว่า 2 พันล้านดอลลาร์ ถือว่าเล็ก/เสี่ยง
        
        # 2. Beta (ความผันผวน)
        beta = info.get('beta', 1.0)
        is_volatile = beta > 1.5 # ผันผวนกว่าตลาด 1.5 เท่า
        
        # 3. Description
        name = info.get('longName', ticker)
        sector = info.get('sector', 'Unknown')
        
        risk_level = "ปลอดภัย ✅"
        warning_msg = ""
        
        if is_small_cap and is_volatile:
            risk_level = "อันตรายมาก 💀"
            warning_msg = "⚠️ เตือนเพื่อน: หุ้นตัวนี้ตัวเล็กและเหวี่ยงแรงมาก เหมือนนั่งรถไฟเหาะ ไม่แนะนำถ้าเงินเย็นไม่พอ!"
        elif is_volatile:
            risk_level = "ผันผวนสูง ⚡"
            warning_msg = "⚠️ เตือน: ราคาขึ้นลงแรง ใจต้องนิ่งนะ"
            
        return {"name": name, "sector": sector, "risk": risk_level, "warning": warning_msg}
    except:
        return {"name": ticker, "sector": "-", "risk": "Unkown", "warning": ""}

def get_insider_activity(ticker):
    try:
        stock = yf.Ticker(ticker)
        insider = stock.insider_transactions
        if insider is None or insider.empty: return "ไม่มีข้อมูล (ผู้บริหารนิ่ง)"
        
        # ตรวจสอบการเทขาย
        sell_count = 0
        details = []
        for i, r in insider.head(5).iterrows():
            trans = str(r.get('Transaction', '')).lower()
            if "sale" in trans: sell_count += 1
            details.append(f"- {r.get('Insider','?')} ทำรายการ {r.get('Transaction','?')}")
            
        summary = "\n".join(details[:3])
        if sell_count >= 2: return f"🚨 ผู้บริหารเริ่มเทขายของ!\n{summary}"
        return f"ปกติ (มีการซื้อขายบ้าง)\n{summary}"
    except: return "N/A"

def get_news_sentiment(ticker):
    try:
        if not GEMINI_API_KEY: return 0, "No API Key"
        stock = yf.Ticker(ticker)
        news = stock.news[:3]
        if not news: return 0, "ไม่มีข่าวใหม่"
        
        headlines = "\n".join([f"- {n['title']}" for n in news])
        
        # ให้ Gemini อ่านข่าวแล้วสรุปอารมณ์
        prompt = f"""
        Read these news headlines for {ticker}:
        {headlines}
        
        Rate the sentiment from -1.0 (Bad) to 1.0 (Good). Just the number.
        """
        response = model.generate_content(prompt)
        return float(response.text.strip()), headlines
    except: return 0, "News Error"

# ==============================================================================
# 🧠 GENIUS ENGINE (ALPHA LOGIC v2)
# ==============================================================================
def analyze_market(ticker):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    try:
        # Data
        df = yf.download(ticker, start=START_DATE, progress=False)
        if len(df) < 100: return None
        curr_price = df['Close'].iloc[-1].item()
        
        # Technicals (RSI)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rsi = 100 - (100 / (1 + (gain / loss)))
        curr_rsi = rsi.iloc[-1].item()
        
        # AI Forecast (LSTM)
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
            ai_model.fit(X, np.array([scaled_data[-1]]), epochs=5, verbose=0)
            ai_model.save(model_file)
            
        pred = ai_model.predict(X, verbose=0)
        pred_price = scaler.inverse_transform(pred)[0][0]
        
        # External Factors
        profile = get_stock_profile(ticker)
        insider_txt = get_insider_activity(ticker)
        news_score, news_txt = get_news_sentiment(ticker)
        
        # --- SCORING SYSTEM ---
        # 1. AI Score (-1 to 1)
        ai_score = np.clip(((pred_price - curr_price)/curr_price)*10, -1, 1)
        
        # 2. RSI Score (Contrarian)
        rsi_score = 1 if curr_rsi < 30 else -1 if curr_rsi > 70 else 0
        
        # 3. Final Hybrid Score
        final_score = (ai_score * 0.4) + (rsi_score * 0.3) + (news_score * 0.3)
        
        # Safety Override (ถ้าหุ้นอันตราย ระบบจะเข้มงวดขึ้น)
        if "อันตราย" in profile['risk']:
            final_score -= 0.3 # หักคะแนนความเสี่ยง
            
        signal = "WAIT ✋"
        color = 0x95a5a6
        if final_score > 0.3: signal = "BUY 🟢"; color = 0x2ecc71
        elif final_score < -0.3: signal = "SELL 🔴"; color = 0xe74c3c
        
        return {
            "price": curr_price, "rsi": curr_rsi, "score": final_score,
            "signal": signal, "color": color, 
            "profile": profile, "insider": insider_txt, "news": news_txt
        }
    except Exception as e:
        print(e)
        return None

# ==============================================================================
# 👨‍🏫 THE MENTOR (GEMINI PERSONA)
# ==============================================================================
def consult_mentor(ticker, data, stats):
    if not GEMINI_API_KEY: return "ระบบพี่เลี้ยงไม่ทำงาน (No API Key)"
    
    prompt = f"""
    Role: คุณคือ "พี่เลี้ยงนักลงทุนระดับโลก" ที่ใจดีและฉลาดมาก หน้าที่คือสอนมือใหม่ (ที่ไม่มีความรู้เลย) ให้เข้าใจง่ายๆ
    
    Topic: วิเคราะห์หุ้น {ticker} ({data['profile']['name']})
    
    Data:
    - สัญญาณระบบ: {data['signal']} (คะแนนความน่าซื้อ {data['score']:.2f}/1.0)
    - ราคา: ${data['price']:.2f}
    - RSI: {data['rsi']:.1f}
    - ความเสี่ยง: {data['profile']['risk']}
    - คำเตือน: {data['profile']['warning']}
    - ผู้บริหาร: {data['insider']}
    - ความแม่นของบอท: {stats}
    
    Task: เขียนคำแนะนำเป็น 3 ส่วน (ใช้ภาษาพูด เป็นกันเอง ใส่ Emoji ได้):
    
    1. 🐣 **ฉบับอนุบาล (ELI5):** เปรียบเทียบสถานการณ์หุ้นตัวนี้กับชีวิตจริง (เช่น เหมือนรถติด, เหมือนของลดราคา, เหมือนวิ่งไล่รถเมล์) เพื่อให้เห็นภาพทันที
    2. 🧠 **ฉบับศิษย์เอก:** อธิบายเหตุผลจริงๆ ว่าทำไมถึงแนะนำแบบนั้น (อ้างอิงกราฟหรือข่าว) แบบสั้นๆ
    3. 🎯 **สรุป:** บอกเพื่อนว่า "ทำยังไงต่อดี?" (ซื้อเลย / รอก่อน / หนีไป)
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except: return "พี่เลี้ยงกำลังจิบกาแฟครับ (Gemini Error)"

# ==============================================================================
# 🎮 DISCORD INTERFACE
# ==============================================================================
@bot.event
async def on_ready(): print(f'🤖 Genius Mentor Online: {bot.user}')

@bot.command()
async def teach(ctx, term: str = None):
    """🎓 สอนคำศัพท์หุ้นแบบเข้าใจง่ายๆ"""
    if not term:
        await ctx.send("❓ อยากให้สอนคำไหนพิมพ์มาเลยครับ เช่น `!teach RSI` หรือ `!teach หุ้นปั่น`")
        return
        
    prompt = f"Explain the investing term '{term}' to a complete beginner using a funny or simple analogy. Keep it short."
    try:
        if GEMINI_API_KEY:
            res = model.generate_content(prompt)
            await ctx.send(f"🎓 **ห้องเรียนหุ้น:**\n{res.text}")
        else: await ctx.send("ไม่มีคีย์อาจารย์ใหญ่ครับ (Gemini Key)")
    except: await ctx.send("อาจารย์ไม่อยู่ครับ")

@bot.command()
async def check(ctx, ticker: str):
    t = ticker.upper()
    msg = await ctx.send(f"👨‍🏫 **ครูกำลังตรวจการบ้านหุ้น {t} ให้ครับ... รอแป๊บนะ**")
    
    loop = asyncio.get_running_loop()
    
    # 1. วิเคราะห์ด้วยคณิตศาสตร์
    data = await loop.run_in_executor(None, analyze_market, t)
    if not data: await msg.edit(content="❌ ครูหาหุ้นตัวนี้ไม่เจอครับ พิมพ์ชื่อถูกไหม?"); return
    
    # 2. เก็บสถิติ
    stats = get_learning_stats(t)
    log_signal(t, data['signal'], data['price'])
    
    # 3. ให้พี่เลี้ยง (Gemini) เรียบเรียงคำพูด
    advice = await loop.run_in_executor(None, consult_mentor, t, data, stats)
    
    # 4. แสดงผล
    embed = discord.Embed(title=f"📘 รายงานผล: {t} ({data['profile']['name']})", color=data['color'])
    
    # Header: สรุปสถานะ
    embed.add_field(name="สัญญาณวันนี้", value=f"**{data['signal']}**", inline=True)
    embed.add_field(name="ความเสี่ยง", value=f"**{data['profile']['risk']}**", inline=True)
    embed.add_field(name="ความแม่นบอท", value=f"{stats}", inline=True)
    
    # Warning Box (ถ้ามี)
    if data['profile']['warning']:
        embed.add_field(name="🚨 **คำเตือนจากครู**", value=data['profile']['warning'], inline=False)
    
    # Body: คำสอนจาก Gemini
    embed.description = f"{advice}"
    
    # Footer: ข้อมูลดิบ (เผื่ออยากดู)
    footer_txt = f"ราคา: ${data['price']:.2f} | RSI: {data['rsi']:.1f} | AI Score: {data['score']:.2f}"
    embed.set_footer(text=footer_txt)
    
    await ctx.send(embed=embed)
    await msg.delete()

@bot.command()
async def add(ctx, *tickers):
    uid = str(ctx.author.id); port = load_json(PORTFOLIOS_FILE); uport = port.get(uid, [])
    uport.extend([t.upper() for t in tickers if t.upper() not in uport])
    port[uid] = uport; save_json(PORTFOLIOS_FILE, port)
    await ctx.send(f"✅ จด {', '.join(tickers)} ลงสมุดพกให้แล้วครับ")

@bot.command()
async def port(ctx):
    uid = str(ctx.author.id); uport = load_json(PORTFOLIOS_FILE).get(uid, [])
    if not uport: await ctx.send("📭 สมุดพกยังว่างอยู่เลยครับ (ใช้ `!add ชื่อหุ้น` เพื่อเพิ่ม)"); return
    
    await ctx.send(f"🚀 **ตรวจสุขภาพพอร์ตของคุณ {ctx.author.name}...**")
    for t in uport:
        loop = asyncio.get_running_loop()
        d = await loop.run_in_executor(None, analyze_market, t)
        if d:
            embed = discord.Embed(title=f"{t} : {d['signal']}", color=d['color'])
            embed.description = f"ความเสี่ยง: {d['profile']['risk']}"
            if d['profile']['warning']: embed.description += f"\n⚠️ {d['profile']['warning']}"
            await ctx.send(embed=embed)

if __name__ == "__main__":
    keep_alive()
    if DISCORD_TOKEN: bot.run(DISCORD_TOKEN)

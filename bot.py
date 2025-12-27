# ==============================================================================
# 🤖 PROJECT: SMART INVESTOR GUARDIAN (SELF-LEARNING EDITION)
# ==============================================================================

import os
import json
import discord
from discord.ext import commands
import asyncio
from flask import Flask
from threading import Thread
from textblob import TextBlob
from datetime import datetime

# --- 🌐 WEB SERVER ---
app = Flask('')

@app.route('/')
def home():
    return "SYSTEM ONLINE: Self-Learning Engine Active."

def run():
    app.run(host='0.0.0.0', port=8080)

def keep_alive():
    t = Thread(target=run)
    t.start()

# --- 🔑 CONFIGURATION ---
DISCORD_BOT_TOKEN = os.environ.get('DISCORD_TOKEN')
PORTFOLIOS_FILE = 'user_portfolios.json'
HISTORY_FILE = 'prediction_history.json' # 📝 สมุดบันทึกผลการทาย
START_DATE = '2020-01-01'
PREDICTION_DAYS = 60

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

# ==============================================================================
# 💾 DATABASE & HISTORY SYSTEM
# ==============================================================================
def load_json(filename):
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                return json.load(f)
        except: return {}
    return {}

def save_json(filename, data):
    with open(filename, 'w') as f:
        json.dump(data, f)

def log_prediction(ticker, signal, price):
    """จดบันทึกการทาย: วันที่, หุ้น, ทายว่าอะไร, ราคาตอนนั้น"""
    history = load_json(HISTORY_FILE)
    if ticker not in history: history[ticker] = []
    
    record = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "signal": signal,
        "entry_price": price,
        "status": "PENDING" # รอตรวจคำตอบ
    }
    history[ticker].append(record)
    save_json(HISTORY_FILE, history)

def get_accuracy_stats(ticker):
    """คำนวณความแม่นยำ"""
    history = load_json(HISTORY_FILE)
    records = history.get(ticker, [])
    if not records: return "N/A"
    
    correct = sum(1 for r in records if r.get('status') == 'CORRECT')
    wrong = sum(1 for r in records if r.get('status') == 'WRONG')
    total = correct + wrong
    
    if total == 0: return "Waiting for results..."
    win_rate = (correct / total) * 100
    return f"🏆 Win Rate: {win_rate:.1f}% ({correct}/{total})"

# ==============================================================================
# 🧠 HYBRID & SELF-LEARNING ENGINE
# ==============================================================================

def analyze_sentiment(ticker):
    import yfinance as yf
    try:
        stock = yf.Ticker(ticker)
        news_list = stock.news
        if not news_list: return 0, []
        
        total_polarity = 0
        headlines = []
        count = 0
        
        for n in news_list[:3]:
            analysis = TextBlob(n['title'])
            score = analysis.sentiment.polarity
            total_polarity += score
            count += 1
            icon = "🟢" if score > 0.1 else "🔴" if score < -0.1 else "😐"
            headlines.append(f"{icon} [{n['title']}]({n['link']})")

        return (total_polarity / count if count > 0 else 0), headlines
    except: return 0, []

def analyze_hybrid(ticker, force_retrain=False):
    """
    Hybrid System + Self Learning
    force_retrain=True คือสั่งให้ 'ลบสมองเก่า' แล้วเรียนรู้ใหม่ทันที
    """
    import numpy as np
    import pandas as pd
    import yfinance as yf
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Dense, LSTM, Dropout
    import os
    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    try:
        # 1. ดึงข้อมูลล่าสุด
        df = yf.download(ticker, start=START_DATE, progress=False)
        if len(df) < 100: return None
        
        curr_price = df['Close'].iloc[-1].item()

        # --- 🔁 SELF-LEARNING CHECK (ตรวจการบ้าน) ---
        # เช็คว่าที่เคยทายไว้ ถูกหรือผิด?
        history = load_json(HISTORY_FILE)
        if ticker in history:
            changed = False
            for record in history[ticker]:
                if record['status'] == 'PENDING':
                    # ถ้าทาย Buy แล้วราคาขึ้น > 1% = ถูก
                    if record['signal'] == 'BUY NOW 🟢' and curr_price > record['entry_price'] * 1.01:
                        record['status'] = 'CORRECT'
                        changed = True
                    # ถ้าทาย Buy แล้วราคาลง = ผิด
                    elif record['signal'] == 'BUY NOW 🟢' and curr_price < record['entry_price']:
                        record['status'] = 'WRONG'
                        changed = True
                    # (Logic เดียวกันสำหรับ Sell)
            if changed: 
                save_json(HISTORY_FILE, history)
                # ถ้ามีผิดเยอะๆ บอทจะตัดสินใจ Retrain เองได้ในอนาคต

        # --- AI & RETRAINING ---
        model_file = f'brain_{ticker}.keras'
        
        # ถ้าสั่ง Retrain หรือหาไฟล์ไม่เจอ -> สร้างใหม่
        if force_retrain and os.path.exists(model_file):
            os.remove(model_file)
            print(f"♻️ Retraining brain for {ticker}...")

        # ... (ส่วนคำนวณ AI และ RSI เหมือนเดิม) ...
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / loss)))
        df = df.dropna()
        curr_rsi = df['RSI'].iloc[-1].item()
        
        features = ['Close', 'RSI']
        dataset = df[features].values
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)
        
        X, y = [], []
        # ใช้ข้อมูลทั้งหมดที่มีจนถึงวันนี้ เพื่อเทรนให้ฉลาดที่สุด
        for i in range(PREDICTION_DAYS, len(scaled_data)):
            X.append(scaled_data[i-PREDICTION_DAYS:i])
            y.append(scaled_data[i, 0])
        X, y = np.array(X), np.array(y)
        
        if os.path.exists(model_file):
            model = load_model(model_file)
        else:
            # สร้างโมเดลใหม่ (นี่คือขั้นตอน Learning)
            model = Sequential([
                LSTM(50, return_sequences=False, input_shape=(PREDICTION_DAYS, 2)),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')
            model.fit(X, y, epochs=10, batch_size=32, verbose=0)
            model.save(model_file)
            
        last_sequence = scaled_data[-PREDICTION_DAYS:].reshape(1, PREDICTION_DAYS, 2)
        pred_scaled = model.predict(last_sequence, verbose=0)
        temp_matrix = np.zeros((1, 2))
        temp_matrix[:, 0] = pred_scaled.flatten()
        pred_price = scaler.inverse_transform(temp_matrix)[:, 0][0]
        
        # --- HYBRID CALCULATION ---
        news_score, headlines = analyze_sentiment(ticker)
        
        ai_score = np.clip(((pred_price - curr_price) / curr_price) * 10, -1, 1)
        rsi_score = 1 if curr_rsi < 30 else -1 if curr_rsi > 70 else 0
        final_score = (ai_score * 0.4) + (rsi_score * 0.3) + (news_score * 0.3)
        
        signal = "HOLD ✋"
        color = 0x95a5a6
        if final_score > 0.25: 
            signal = "BUY NOW 🟢"
            color = 0x2ecc71
        elif final_score < -0.25: 
            signal = "SELL NOW 🔴"
            color = 0xe74c3c

        # 📝 บันทึกผลการทำนายลงสมุดทันที
        log_prediction(ticker, signal, curr_price)

        stats = get_accuracy_stats(ticker)
        
        return {
            "price": curr_price, "fair": pred_price, "rsi": curr_rsi,
            "score": final_score, "signal": signal, "color": color,
            "news": headlines, "stats": stats
        }
    except Exception as e:
        print(f"Error: {e}")
        return None

# ==============================================================================
# 🎮 DISCORD COMMANDS
# ==============================================================================
@bot.event
async def on_ready():
    print(f'🤖 Bot Online: {bot.user}')

@bot.command()
async def add(ctx, *tickers):
    """เพิ่มหุ้น"""
    user_id = ctx.author.id
    port = load_json(PORTFOLIOS_FILE)
    user_port = port.get(str(user_id), [])
    
    added = []
    for t in tickers:
        t = t.upper().replace(",", "")
        if t not in user_port:
            user_port.append(t)
            added.append(t)
    
    port[str(user_id)] = user_port
    save_json(PORTFOLIOS_FILE, port)
    if added: await ctx.send(f"✅ Added: {', '.join(added)}")

@bot.command()
async def remove(ctx, *tickers):
    """ลบหุ้น"""
    user_id = ctx.author.id
    port = load_json(PORTFOLIOS_FILE)
    user_port = port.get(str(user_id), [])
    removed = []
    for t in tickers:
        t = t.upper().replace(",", "")
        if t in user_port:
            user_port.remove(t)
            removed.append(t)
    port[str(user_id)] = user_port
    save_json(PORTFOLIOS_FILE, port)
    await ctx.send(f"🗑️ Removed: {', '.join(removed)}")

@bot.command()
async def port(ctx):
    """เช็คพอร์ต"""
    user_id = ctx.author.id
    port = load_json(PORTFOLIOS_FILE)
    user_port = port.get(str(user_id), [])
    
    if not user_port:
        await ctx.send("📭 Portfolio is empty.")
        return

    await ctx.send(f"🚀 Analyzing portfolio for {ctx.author.name}...")
    for ticker in user_port:
        loop = asyncio.get_running_loop()
        res = await loop.run_in_executor(None, analyze_hybrid, ticker, False)
        if res:
            embed = discord.Embed(title=f"📊 {ticker}", color=res['color'])
            embed.add_field(name="Price", value=f"${res['price']:.2f}", inline=True)
            embed.add_field(name="Signal", value=f"**{res['signal']}**", inline=True)
            embed.add_field(name="Accuracy", value=res['stats'], inline=True)
            await ctx.send(embed=embed)

@bot.command()
async def check(ctx, ticker: str):
    """เช็คหุ้นรายตัว"""
    ticker = ticker.upper()
    msg = await ctx.send(f"🔄 Analyzing **{ticker}**...")
    loop = asyncio.get_running_loop()
    res = await loop.run_in_executor(None, analyze_hybrid, ticker, False)
    
    if res:
        embed = discord.Embed(title=f"💎 Analysis: {ticker}", color=res['color'])
        embed.add_field(name="Price", value=f"${res['price']:.2f}", inline=True)
        embed.add_field(name="Fair Price", value=f"${res['fair']:.2f}", inline=True)
        embed.add_field(name="RSI", value=f"{res['rsi']:.1f}", inline=True)
        embed.add_field(name="Hybrid Score", value=f"**{res['score']:.2f}**", inline=True)
        embed.add_field(name="Strategy", value=f"**{res['signal']}**", inline=False)
        embed.add_field(name="🏆 Past Accuracy", value=res['stats'], inline=False)
        
        if res['news']:
            embed.add_field(name="📰 News", value="\n".join(res['news']), inline=False)
        await ctx.send(embed=embed)
        await msg.delete()

@bot.command()
async def retrain(ctx, ticker: str):
    """🧠 สั่งให้บอทลืมความรู้เก่า และเทรนใหม่เดี๋ยวนี้"""
    ticker = ticker.upper()
    msg = await ctx.send(f"🧠 **Retraining Mode:** กำลังล้างสมองและเรียนรู้กราฟล่าสุดของ {ticker}...")
    
    loop = asyncio.get_running_loop()
    # ส่ง True ไปเพื่อบังคับลบไฟล์โมเดล
    res = await loop.run_in_executor(None, analyze_hybrid, ticker, True)
    
    if res:
        await msg.delete()
        await ctx.send(f"✅ **Retrain Complete!** บอทฉลาดขึ้นแล้วครับสำหรับหุ้น {ticker}")
        # แสดงผลลัพธ์หลังเทรนเสร็จทันที
        await check(ctx, ticker)

# --- STARTUP ---
if __name__ == "__main__":
    keep_alive()
    if DISCORD_BOT_TOKEN:
        bot.run(DISCORD_BOT_TOKEN)

# ==============================================================================
# 🤖 PROJECT: SMART INVESTOR GUARDIAN (ULTIMATE HYBRID EDITION)
# ==============================================================================

import os
import json
import discord
from discord.ext import commands
import asyncio
from flask import Flask
from threading import Thread
from textblob import TextBlob # สมองส่วนภาษา (Sentiment)

# --- 🌐 WEB SERVER (Keep Alive) ---
app = Flask('')

@app.route('/')
def home():
    return "SYSTEM ONLINE: Hybrid Engine Running."

def run():
    app.run(host='0.0.0.0', port=8080)

def keep_alive():
    t = Thread(target=run)
    t.start()

# --- 🔑 CONFIGURATION ---
DISCORD_BOT_TOKEN = os.environ.get('DISCORD_TOKEN')
PORTFOLIOS_FILE = 'user_portfolios.json'
START_DATE = '2020-01-01'
PREDICTION_DAYS = 60

# ตั้งค่าบอท
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

# ==============================================================================
# 💾 DATABASE SYSTEM
# ==============================================================================
def load_data():
    if os.path.exists(PORTFOLIOS_FILE):
        try:
            with open(PORTFOLIOS_FILE, 'r') as f:
                return json.load(f)
        except: return {}
    return {}

def save_data(data):
    with open(PORTFOLIOS_FILE, 'w') as f:
        json.dump(data, f)

def get_user_portfolio(user_id):
    data = load_data()
    return data.get(str(user_id), [])

def update_user_portfolio(user_id, portfolio):
    data = load_data()
    data[str(user_id)] = portfolio
    save_data(data)

# ==============================================================================
# 🧠 HYBRID ENGINE (AI + TECHNICAL + NEWS SENTIMENT)
# ==============================================================================

def analyze_sentiment(ticker):
    """วิเคราะห์อารมณ์ข่าว (Sentiment Analysis)"""
    import yfinance as yf
    try:
        stock = yf.Ticker(ticker)
        news_list = stock.news
        
        if not news_list:
            return 0, [] # ไม่มีข่าว = คะแนนเป็นกลาง (0)

        total_polarity = 0
        headlines = []
        count = 0

        # วิเคราะห์ 5 ข่าวล่าสุด
        for n in news_list[:5]:
            title = n['title']
            link = n['link']
            
            # ใช้ TextBlob วิเคราะห์ประโยค (-1 ถึง +1)
            analysis = TextBlob(title)
            score = analysis.sentiment.polarity
            total_polarity += score
            count += 1
            
            # เก็บหัวข้อข่าวไว้แสดงผล
            icon = "😐"
            if score > 0.1: icon = "🟢"
            elif score < -0.1: icon = "🔴"
            headlines.append(f"{icon} [{title}]({link})")

        avg_score = total_polarity / count if count > 0 else 0
        return avg_score, headlines
    except:
        return 0, []

def analyze_hybrid(ticker):
    """ระบบลูกผสม: AI (40%) + RSI (30%) + News (30%)"""
    # Lazy Import
    import numpy as np
    import pandas as pd
    import yfinance as yf
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Dense, LSTM, Dropout
    import os
    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    try:
        # --- PART 1: TECHNICAL & AI ---
        df = yf.download(ticker, start=START_DATE, progress=False)
        if len(df) < 100: return None
        
        # RSI Calculation
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / loss)))
        df = df.dropna()
        curr_rsi = df['RSI'].iloc[-1].item()
        
        # AI Prediction
        features = ['Close', 'RSI']
        dataset = df[features].values
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)
        
        X = []
        # สร้างข้อมูลทดสอบชุดสุดท้าย
        last_sequence = scaled_data[-PREDICTION_DAYS:].reshape(1, PREDICTION_DAYS, 2)
        
        # Load/Train Model (Simplified for speed)
        model_file = f'brain_{ticker}.keras'
        if os.path.exists(model_file):
            model = load_model(model_file)
        else:
            # สร้างโมเดลแบบรวดเร็วถ้าไม่เจอไฟล์
            X_train, y_train = [], []
            for i in range(PREDICTION_DAYS, len(scaled_data)):
                X_train.append(scaled_data[i-PREDICTION_DAYS:i])
                y_train.append(scaled_data[i, 0])
            X_train, y_train = np.array(X_train), np.array(y_train)
            
            model = Sequential([
                LSTM(50, return_sequences=False, input_shape=(PREDICTION_DAYS, 2)),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')
            model.fit(X_train, y_train, epochs=5, verbose=0) # เทรนเร็วๆ
            model.save(model_file)
            
        pred_scaled = model.predict(last_sequence, verbose=0)
        temp_matrix = np.zeros((1, 2))
        temp_matrix[:, 0] = pred_scaled.flatten()
        pred_price = scaler.inverse_transform(temp_matrix)[:, 0][0]
        curr_price = df['Close'].iloc[-1].item()
        
        # --- PART 2: NEWS SENTIMENT ---
        news_score, headlines = analyze_sentiment(ticker) # ได้ค่า -1 ถึง 1
        
        # --- PART 3: WEIGHTED CALCULATION (สูตรลับของเรา) ---
        
        # 1. AI Score (-1 ถึง 1): ถ้า Fair Price > Current Price = เป็นบวก
        ai_upside = (pred_price - curr_price) / curr_price
        ai_score = np.clip(ai_upside * 10, -1, 1) # ขยายสเกลเพื่อให้มีน้ำหนัก
        
        # 2. RSI Score (-1 ถึง 1): RSI ต่ำ (Oversold) = ดี (+), RSI สูง = แย่ (-)
        rsi_score = 0
        if curr_rsi < 30: rsi_score = 1
        elif curr_rsi > 70: rsi_score = -1
        else: rsi_score = 0.5 - (curr_rsi / 100) # ค่ากลางๆ
        
        # 3. Final Hybrid Score
        # สูตร: (AI * 0.4) + (RSI * 0.3) + (News * 0.3)
        final_score = (ai_score * 0.4) + (rsi_score * 0.3) + (news_score * 0.3)
        
        # แปลผลคะแนน
        signal = "HOLD ✋"
        color = 0x95a5a6
        advice = "ปัจจัยต่างๆ ยังขัดแย้งกัน ชะลอการลงทุนครับ"
        
        if final_score > 0.25:
            signal = "STRONG BUY 🟢"
            color = 0x2ecc71
            advice = f"สัญญาณดีทุกด้าน! AI มองขึ้น ({ai_score:.2f}), RSI สวย, ข่าวเป็นใจ ({news_score:.2f})"
        elif final_score < -0.25:
            signal = "STRONG SELL 🔴"
            color = 0xe74c3c
            advice = f"สัญญาณอันตราย! AI มองลง ({ai_score:.2f}), หรือข่าวลบแรงมาก ({news_score:.2f})"
            
        return {
            "price": curr_price,
            "fair": pred_price,
            "rsi": curr_rsi,
            "news_headlines": headlines,
            "news_score": news_score,
            "final_score": final_score,
            "signal": signal,
            "color": color,
            "advice": advice
        }
        
    except Exception as e:
        print(f"Error: {e}")
        return None

# ==============================================================================
# 🎮 DISCORD COMMANDS
# ==============================================================================
@bot.event
async def on_ready():
    print(f'🤖 Hybrid Bot Online: {bot.user}')

@bot.command()
async def add(ctx, *tickers):
    """เพิ่มหุ้นเข้าพอร์ตส่วนตัว"""
    user_id = ctx.author.id
    portfolio = get_user_portfolio(user_id)
    added = []
    
    # Check Valid Ticker
    import yfinance as yf
    msg = await ctx.send("🔍 Verifying tickers...")
    
    for ticker in tickers:
        ticker = ticker.upper().replace(",", "")
        if ticker not in portfolio:
            try:
                if not yf.Ticker(ticker).history(period="1d").empty:
                    portfolio.append(ticker)
                    added.append(ticker)
            except: pass
            
    update_user_portfolio(user_id, portfolio)
    await msg.delete()
    if added: await ctx.send(f"✅ Added to **{ctx.author.name}'s** portfolio: {', '.join(added)}")
    else: await ctx.send("❌ No new tickers added.")

@bot.command()
async def remove(ctx, *tickers):
    user_id = ctx.author.id
    portfolio = get_user_portfolio(user_id)
    removed = []
    for ticker in tickers:
        ticker = ticker.upper().replace(",", "")
        if ticker in portfolio:
            portfolio.remove(ticker)
            removed.append(ticker)
    update_user_portfolio(user_id, portfolio)
    await ctx.send(f"🗑️ Removed: {', '.join(removed)}")

@bot.command()
async def port(ctx):
    """เช็คพอร์ตด้วยระบบ Hybrid"""
    user_id = ctx.author.id
    portfolio = get_user_portfolio(user_id)
    
    if not portfolio:
        await ctx.send("📭 Portfolio is empty.")
        return

    await ctx.send(f"🚀 Running **Hybrid Analysis** for {ctx.author.name}...")
    
    for ticker in portfolio:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, analyze_hybrid, ticker)
        
        if result:
            embed = discord.Embed(title=f"📊 {ticker}", color=result['color'])
            embed.add_field(name="Price", value=f"${result['price']:.2f}", inline=True)
            embed.add_field(name="Score", value=f"{result['final_score']:.2f} / 1.0", inline=True)
            embed.add_field(name="Signal", value=f"**{result['signal']}**", inline=True)
            embed.add_field(name="💡 AI Advice", value=result['advice'], inline=False)
            await ctx.send(embed=embed)

@bot.command()
async def check(ctx, ticker: str):
    """เจาะลึกรายตัว (AI + News)"""
    ticker = ticker.upper()
    msg = await ctx.send(f"🔄 Analyzing **{ticker}** (AI + Tech + News)...")
    
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, analyze_hybrid, ticker)
    
    if result:
        embed = discord.Embed(title=f"💎 Hybrid Analysis: {ticker}", color=result['color'])
        
        # Section 1: Numbers
        embed.add_field(name="Market Price", value=f"${result['price']:.2f}", inline=True)
        embed.add_field(name="AI Fair Price", value=f"${result['fair']:.2f}", inline=True)
        embed.add_field(name="RSI", value=f"{result['rsi']:.1f}", inline=True)
        
        # Section 2: Scores
        embed.add_field(name="News Sentiment", value=f"{result['news_score']:.2f}", inline=True)
        embed.add_field(name="Hybrid Score", value=f"**{result['final_score']:.2f}**", inline=True)
        embed.add_field(name="Strategy", value=f"**{result['signal']}**", inline=True)
        
        # Section 3: Advice & News
        embed.add_field(name="💡 Summary", value=result['advice'], inline=False)
        
        if result['news_headlines']:
            news_text = "\n".join(result['news_headlines'][:3])
            embed.add_field(name="📰 Top News & Sentiment", value=news_text, inline=False)
            
        await ctx.send(embed=embed)
        await msg.delete()
    else:
        await ctx.send(f"❌ Could not analyze {ticker}")

# --- STARTUP ---
if __name__ == "__main__":
    keep_alive()
    if DISCORD_BOT_TOKEN:
        bot.run(DISCORD_BOT_TOKEN)

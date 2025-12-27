# ==============================================================================
# 🤖 PROJECT: SMART INVESTOR GUARDIAN (PHASE 2: COMMUNITY & NEWS)
# ==============================================================================

import os
import json
import discord
from discord.ext import commands
import asyncio
from flask import Flask
from threading import Thread

# --- 🌐 WEB SERVER (Keep Alive) ---
app = Flask('')

@app.route('/')
def home():
    return "I'm alive! Smart Investor Guardian is running."

def run():
    app.run(host='0.0.0.0', port=8080)

def keep_alive():
    t = Thread(target=run)
    t.start()

# --- 🔑 CONFIGURATION ---
DISCORD_BOT_TOKEN = os.environ.get('DISCORD_TOKEN')
PORTFOLIOS_FILE = 'user_portfolios.json' # เปลี่ยนชื่อไฟล์ให้สื่อความหมาย
START_DATE = '2020-01-01'
PREDICTION_DAYS = 60

# ตั้งค่าบอท
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

# ==============================================================================
# 💾 DATABASE SYSTEM (ระบบแยกพอร์ตรายคน)
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
# 🧠 AI & NEWS ENGINE
# ==============================================================================
def get_stock_news(ticker):
    import yfinance as yf
    try:
        stock = yf.Ticker(ticker)
        news_list = stock.news
        if not news_list: return None
        
        # ดึง 3 ข่าวล่าสุด
        headlines = []
        for n in news_list[:3]:
            headlines.append(f"📰 [{n['title']}]({n['link']})")
        return headlines
    except:
        return None

def analyze_stock(ticker):
    # Lazy Import เพื่อความเร็ว
    import numpy as np
    import pandas as pd
    import yfinance as yf
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Dense, LSTM, Dropout
    import os
    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    try:
        # 1. ดึงข้อมูล
        df = yf.download(ticker, start=START_DATE, progress=False)
        if len(df) < 100: return None
        
        # 2. คำนวณ RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / loss)))
        df = df.dropna()
        
        # 3. เตรียม AI
        features = ['Close', 'RSI']
        dataset = df[features].values
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)
        
        X, y = [], []
        for i in range(PREDICTION_DAYS, len(scaled_data)):
            X.append(scaled_data[i-PREDICTION_DAYS:i])
            y.append(scaled_data[i, 0])
        X, y = np.array(X), np.array(y)
        
        # 4. โมเดล
        model_file = f'brain_{ticker}.keras'
        if os.path.exists(model_file):
            model = load_model(model_file)
        else:
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(X.shape[1], 2)),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(X, y, epochs=15, batch_size=32, verbose=0)
            model.save(model_file)
            
        # 5. ทำนาย
        last_sequence = scaled_data[-PREDICTION_DAYS:].reshape(1, PREDICTION_DAYS, 2)
        pred_scaled = model.predict(last_sequence, verbose=0)
        
        temp_matrix = np.zeros((1, 2))
        temp_matrix[:, 0] = pred_scaled.flatten()
        pred_price = scaler.inverse_transform(temp_matrix)[:, 0][0]
        
        curr_price = df['Close'].iloc[-1].item()
        curr_rsi = df['RSI'].iloc[-1].item()
        
        # Logic คำแนะนำ
        bias = curr_price - pred_price
        fair_price = pred_price + (bias * 0.5)
        
        signal = "HOLD ✋"
        color = 0x95a5a6
        advice = "ราคาอยู่ในช่วงทรงตัว ถือต่อได้ครับ"

        if fair_price > curr_price * 1.02 and curr_rsi < 65:
            signal = "BUY NOW 🟢"
            color = 0x2ecc71
            advice = "AI มองว่าราคายังไปได้ต่อ และ RSI ไม่สูงเกินไป น่าสะสมครับ"
        elif fair_price < curr_price * 0.98 or curr_rsi > 75:
            signal = "SELL NOW 🔴"
            color = 0xe74c3c
            advice = "ราคาเริ่มตึงๆ หรือ RSI สูงเกินไป (Overbought) ระวังแรงขายทำกำไรครับ"
            
        return {
            "price": curr_price, 
            "fair": fair_price, 
            "rsi": curr_rsi, 
            "signal": signal, 
            "color": color,
            "advice": advice
        }
    except Exception as e:
        print(e)
        return None

# ==============================================================================
# 🎮 DISCORD COMMANDS
# ==============================================================================
@bot.event
async def on_ready():
    print(f'🤖 Community Bot Online: {bot.user}')

@bot.command()
async def add(ctx, *tickers):
    """เพิ่มหุ้น (ส่วนตัว)"""
    user_id = ctx.author.id # จำ ID ของคนพิมพ์
    portfolio = get_user_portfolio(user_id)
    import yfinance as yf 
    
    added = []
    msg = await ctx.send(f"🔍 กำลังเพิ่มหุ้นเข้าพอร์ตส่วนตัวของคุณ...")
    
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
    
    if added: 
        await ctx.send(f"✅ **{ctx.author.name}** เพิ่มสำเร็จ: {', '.join(added)}")
    else: 
        await ctx.send("❌ ไม่ได้เพิ่มหุ้นใหม่ (หรือชื่อผิด/มีอยู่แล้ว)")

@bot.command()
async def remove(ctx, *tickers):
    """ลบหุ้น (ส่วนตัว)"""
    user_id = ctx.author.id
    portfolio = get_user_portfolio(user_id)
    removed = []
    
    for ticker in tickers:
        ticker = ticker.upper().replace(",", "")
        if ticker in portfolio:
            portfolio.remove(ticker)
            removed.append(ticker)
            
    update_user_portfolio(user_id, portfolio)
    await ctx.send(f"🗑️ ลบจากพอร์ตของ **{ctx.author.name}**: {', '.join(removed)}")

@bot.command()
async def show(ctx):
    """ดูพอร์ต (ส่วนตัว)"""
    user_id = ctx.author.id
    portfolio = get_user_portfolio(user_id)
    if portfolio:
        await ctx.send(f"📋 พอร์ตของ **{ctx.author.name}**: {', '.join(portfolio)}")
    else:
        await ctx.send(f"📭 คุณ **{ctx.author.name}** ยังไม่มีหุ้นในพอร์ตครับ (ใช้ !add)")

@bot.command()
async def port(ctx):
    """วิเคราะห์พอร์ต (ส่วนตัว)"""
    user_id = ctx.author.id
    portfolio = get_user_portfolio(user_id)
    
    if not portfolio:
        await ctx.send("📭 พอร์ตว่างเปล่าครับ")
        return

    await ctx.send(f"🚀 กำลังสแกนพอร์ตของ **{ctx.author.name}** ({len(portfolio)} ตัว)...")
    
    for ticker in portfolio:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, analyze_stock, ticker)
        
        if result:
            embed = discord.Embed(title=f"📊 {ticker}", color=result['color'])
            embed.add_field(name="Price", value=f"${result['price']:.2f}", inline=True)
            embed.add_field(name="Signal", value=f"**{result['signal']}**", inline=True)
            embed.add_field(name="AI Advice", value=result['advice'], inline=False)
            await ctx.send(embed=embed)

@bot.command()
async def news(ctx, ticker: str):
    """อ่านข่าวหุ้น: !news AAPL"""
    ticker = ticker.upper()
    await ctx.send(f"📰 กำลังค้นหาข่าวล่าสุดของ **{ticker}**...")
    
    loop = asyncio.get_running_loop()
    headlines = await loop.run_in_executor(None, get_stock_news, ticker)
    
    if headlines:
        embed = discord.Embed(title=f"News: {ticker}", color=0x3498db)
        embed.description = "\n\n".join(headlines)
        embed.set_footer(text="Source: Yahoo Finance")
        await ctx.send(embed=embed)
    else:
        await ctx.send(f"❌ ไม่พบข่าวล่าสุดของ {ticker}")

@bot.command()
async def check(ctx, ticker: str):
    """เช็คหุ้น + ข่าว"""
    ticker = ticker.upper()
    msg = await ctx.send(f"🔄 กำลังวิเคราะห์ข้อมูลและข่าวสารของ **{ticker}**...")
    
    loop = asyncio.get_running_loop()
    # รัน AI และ ข่าว พร้อมกัน
    ai_task = loop.run_in_executor(None, analyze_stock, ticker)
    news_task = loop.run_in_executor(None, get_stock_news, ticker)
    
    result, headlines = await asyncio.gather(ai_task, news_task)
    
    if result:
        embed = discord.Embed(title=f"💎 Deep Analysis: {ticker}", color=result['color'])
        embed.add_field(name="Market Price", value=f"${result['price']:.2f}", inline=True)
        embed.add_field(name="Fair Price", value=f"${result['fair']:.2f}", inline=True)
        embed.add_field(name="RSI", value=f"{result['rsi']:.1f}", inline=True)
        embed.add_field(name="Strategy", value=f"**{result['signal']}**", inline=False)
        embed.add_field(name="💡 AI Advice", value=result['advice'], inline=False)
        
        if headlines:
            embed.add_field(name="📰 Recent News", value="\n".join(headlines), inline=False)
            
        await ctx.send(embed=embed)
        await msg.delete()
    else:
        await ctx.send(f"❌ ไม่สามารถวิเคราะห์ {ticker} ได้")

# --- STARTUP ---
if __name__ == "__main__":
    keep_alive()
    if DISCORD_BOT_TOKEN:
        bot.run(DISCORD_BOT_TOKEN)

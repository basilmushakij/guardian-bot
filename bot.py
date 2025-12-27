
import os
import json
import discord
from discord.ext import commands
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout
import asyncio

DISCORD_BOT_TOKEN = os.environ.get('DISCORD_TOKEN')
PORTFOLIO_FILE = 'my_portfolio.json'


START_DATE = '2020-01-01'
PREDICTION_DAYS = 60
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

# ==============================================================================
# 💾 DATABASE SYSTEM (ระบบจำรายชื่อหุ้น)
# ==============================================================================
def load_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        with open(PORTFOLIO_FILE, 'r') as f:
            return json.load(f)
    return []

def save_portfolio(portfolio):
    with open(PORTFOLIO_FILE, 'w') as f:
        json.dump(portfolio, f)

# ==============================================================================
# 🧠 AI ENGINE (ระบบสมอง AI)
# ==============================================================================
def analyze_stock(ticker):
    try:
        # 1. ดึงข้อมูล
        df = yf.download(ticker, start=START_DATE, progress=False)
        if len(df) < 100: return None
        
        # 2. คำนวณ Indicator
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / loss)))
        df = df.dropna()
        
        # 3. เตรียมข้อมูล
        features = ['Close', 'RSI']
        dataset = df[features].values
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)
        
        X, y = [], []
        for i in range(PREDICTION_DAYS, len(scaled_data)):
            X.append(scaled_data[i-PREDICTION_DAYS:i])
            y.append(scaled_data[i, 0])
        X, y = np.array(X), np.array(y)
        
        # 4. โหลด/สร้างโมเดล
        model_file = f'brain_{ticker}.keras'
        if os.path.exists(model_file):
            model = load_model(model_file)
        else:
            print(f"⚡ สร้างสมองใหม่ให้ {ticker}...")
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
        
        # Calibration & Logic
        bias = curr_price - pred_price
        fair_price = pred_price + (bias * 0.5)
        
        signal = "HOLD ✋"
        color = 0x95a5a6
        if fair_price > curr_price * 1.02 and curr_rsi < 65:
            signal = "BUY NOW 🟢"
            color = 0x2ecc71
        elif fair_price < curr_price * 0.98 or curr_rsi > 75:
            signal = "SELL NOW 🔴"
            color = 0xe74c3c
            
        return {"price": curr_price, "fair": fair_price, "rsi": curr_rsi, "signal": signal, "color": color}
    except Exception as e:
        print(f"Error {ticker}: {e}")
        return None

# ==============================================================================
# 🎮 DISCORD COMMANDS (คำสั่งบอท)
# ==============================================================================

@bot.event
async def on_ready():
    print('='*40)
    print(f'🤖 บอทออนไลน์แล้ว! ชื่อ: {bot.user}')
    print('พร้อมใช้งานคำสั่ง: !add, !remove, !port, !check')
    print('='*40)

@bot.command()
async def add(ctx, *tickers):
    
    if not tickers:
        await ctx.send("⚠️ กรุณาพิมพ์ชื่อหุ้นด้วยครับ เช่น `!add AAPL MSFT`")
        return

    portfolio = load_portfolio()
    added = []
    exists = []
    not_found = []

    status_msg = await ctx.send(f"🔍 กำลังตรวจสอบรายชื่อหุ้น {len(tickers)} ตัว...")

    for ticker in tickers:
        ticker = ticker.upper().replace(",", "")
        if ticker in portfolio:
            exists.append(ticker)
            continue
        try:
            data = yf.Ticker(ticker).history(period="1d")
            if not data.empty:
                portfolio.append(ticker)
                added.append(ticker)
            else:
                not_found.append(ticker)
        except:
            not_found.append(ticker)

    save_portfolio(portfolio)
    await status_msg.delete()

    response = "📝 **ผลการเพิ่มหุ้น:**\n"
    if added: response += f"✅ **เพิ่มสำเร็จ:** {', '.join(added)}\n"
    if exists: response += f"⚠️ **มีอยู่แล้ว:** {', '.join(exists)}\n"
    if not_found: response += f"❌ **ไม่พบชื่อ:** {', '.join(not_found)}"
    await ctx.send(response)

@bot.command()
async def remove(ctx, *tickers):
   
    if not tickers:
        await ctx.send("⚠️ พิมพ์ชื่อหุ้นที่จะลบด้วยครับ")
        return

    portfolio = load_portfolio()
    removed = []
    
    for ticker in tickers:
        ticker = ticker.upper().replace(",", "")
        if ticker in portfolio:
            portfolio.remove(ticker)
            removed.append(ticker)
            
    save_portfolio(portfolio)
    
    if removed:
        await ctx.send(f"🗑️ ลบออกจากพอร์ตแล้ว: {', '.join(removed)}")
    else:
        await ctx.send("⚠️ ไม่พบหุ้นที่ระบุในพอร์ตครับ")

@bot.command()
async def show(ctx):
    
    portfolio = load_portfolio()
    if portfolio:
        await ctx.send(f"📋 **หุ้นในพอร์ต:** {', '.join(portfolio)}")
    else:
        await ctx.send("📭 พอร์ตว่างเปล่า (ใช้ !add เพื่อเพิ่ม)")

@bot.command()
async def port(ctx):
    
    portfolio = load_portfolio()
    if not portfolio:
        await ctx.send("📭 พอร์ตว่างเปล่าครับ")
        return

    await ctx.send(f"🚀 กำลังสแกนพอร์ต ({len(portfolio)} ตัว)... รอสักครู่")
    
    for ticker in portfolio:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, analyze_stock, ticker)
        
        if result:
            embed = discord.Embed(title=f"📊 {ticker}", color=result['color'])
            embed.add_field(name="Price", value=f"${result['price']:.2f}", inline=True)
            embed.add_field(name="Fair", value=f"${result['fair']:.2f}", inline=True)
            embed.add_field(name="Signal", value=f"**{result['signal']}**", inline=True)
            await ctx.send(embed=embed)
        else:
            await ctx.send(f"❌ เช็ค {ticker} ไม่ได้")

@bot.command()
async def check(ctx, ticker: str):

    ticker = ticker.upper()
    msg = await ctx.send(f"🔄 วิเคราะห์ **{ticker}**...")
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, analyze_stock, ticker)
    
    if result:
        embed = discord.Embed(title=f"Analysis: {ticker}", color=result['color'])
        embed.add_field(name="Price", value=f"${result['price']:.2f}", inline=True)
        embed.add_field(name="Fair Price", value=f"${result['fair']:.2f}", inline=True)
        embed.add_field(name="RSI", value=f"{result['rsi']:.1f}", inline=True)
        embed.add_field(name="Strategy", value=f"**{result['signal']}**", inline=False)
        await ctx.send(embed=embed)
        await msg.delete()
    else:
        await ctx.send(f"❌ ไม่พบข้อมูล {ticker}")


if DISCORD_BOT_TOKEN == 'ใส่_BOT_TOKEN_ของคุณที่นี่':
    print("❌ ERROR: ลืมใส่ Token ครับ!")
else:
    bot.run(DISCORD_BOT_TOKEN)
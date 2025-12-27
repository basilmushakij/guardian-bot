# ==============================================================================
# 🤖 PROJECT: SMART INVESTOR GUARDIAN (THE JUDGE EDITION)
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

# --- 🌐 WEB SERVER (Keep Alive) ---
app = Flask('')
@app.route('/')
def home(): return "SYSTEM ONLINE: The Judge is watching."
def run(): app.run(host='0.0.0.0', port=8080)
def keep_alive(): t = Thread(target=run); t.start()

# --- 🔑 CONFIGURATION ---
DISCORD_TOKEN = os.environ.get('DISCORD_TOKEN')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY') # ดึงคีย์จาก Render
PORTFOLIOS_FILE = 'user_portfolios.json'
START_DATE = '2020-01-01'
PREDICTION_DAYS = 60

# ตั้งค่า Gemini
try:
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-pro')
        print("✅ Gemini AI Connected")
    else:
        print("⚠️ Warning: GEMINI_API_KEY not found")
except Exception as e:
    print(f"⚠️ Gemini Error: {e}")

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

# ==============================================================================
# 🕵️ INSIDER & DATA GATHERING (นักสืบ)
# ==============================================================================
def get_insider_activity(ticker):
    """สืบข้อมูลผู้บริหาร: มีการเทขายหรือซื้อเก็บไหม?"""
    import yfinance as yf
    try:
        stock = yf.Ticker(ticker)
        insider = stock.insider_transactions
        if insider is None or insider.empty:
            return "ไม่พบข้อมูล (หรือตลาดไม่เปิดเผย)"
        
        latest = insider.head(5)
        summary = ""
        for index, row in latest.iterrows():
            date_str = str(index)[:10] 
            text = f"- {date_str}: {row.get('Insider', 'Unknown')} ทำรายการ {row.get('Transaction', 'Unknown')} {row.get('Shares', 0)} หุ้น"
            summary += text + "\n"
        return summary
    except: return "ไม่สามารถดึงข้อมูลได้"

def get_technical_data(ticker):
    """คำนวณกราฟและ AI (ฝ่ายเทคนิค)"""
    import yfinance as yf
    import numpy as np
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
        
        # Simple AI Prediction
        data = df[['Close']].values
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)
        
        X = []
        X.append(scaled_data[-PREDICTION_DAYS:])
        X = np.array(X)
        
        # Load/Create Model (Fast Mode)
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
        
        trend = "ขาขึ้น (UP)" if pred_price > curr_price else "ขาลง (DOWN)"
        
        return {"price": curr_price, "ai_price": pred_price, "rsi": curr_rsi, "trend": trend}
    except Exception as e:
        print(f"Tech Error: {e}")
        return None

def get_news_summary(ticker):
    import yfinance as yf
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        headlines = [f"- {n['title']}" for n in news[:5]]
        return "\n".join(headlines)
    except: return "ไม่มีข่าวล่าสุด"

# ==============================================================================
# ⚖️ THE JUDGE (GEMINI ตัดสินความจริง)
# ==============================================================================
def consult_the_judge(ticker, tech_data, insider_data, news_data):
    """ส่งข้อมูลให้ Gemini วิเคราะห์"""
    if not GEMINI_API_KEY:
        return "⚠️ ไม่พบ Gemini API Key กรุณาตั้งค่าใน Render"

    prompt = f"""
    คุณคือ "The Judge" AI นักลงทุนระดับโลก. วิเคราะห์หุ้น {ticker} จากข้อมูลนี้ แล้วให้คำแนะนำสั้นๆเป็นภาษาไทย:
    
    1. ข้อมูลเทคนิค: ราคา ${tech_data['price']:.2f}, AIทำนาย: ${tech_data['ai_price']:.2f} ({tech_data['trend']}), RSI: {tech_data['rsi']:.1f}
    2. ธุรกรรมผู้บริหาร:
    {insider_data}
    3. ข่าวล่าสุด:
    {news_data}
    
    ตอบ 3 ข้อ:
    1. สรุปสถานการณ์ (สั้นๆ)
    2. ความเสี่ยง (เช่น ผู้บริหารขายของ หรือ RSI สูงเกิน)
    3. ฟันธง: ซื้อ / ขาย / หรือ ถือรอ (พร้อมเหตุผล 1 ประโยค)
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"เกิดข้อผิดพลาดในการเรียกผู้พิพากษา: {e}"

# ==============================================================================
# 🎮 COMMANDS
# ==============================================================================
@bot.event
async def on_ready(): print(f'🤖 The Judge is Online: {bot.user}')

@bot.command()
async def check(ctx, ticker: str):
    ticker = ticker.upper()
    msg = await ctx.send(f"⚖️ **กำลังเปิดศาลไต่สวนคดี {ticker}...**\n(รวบรวมหลักฐาน + เรียกพยาน...)")
    
    loop = asyncio.get_running_loop()
    
    # 1. รวบรวมหลักฐาน
    tech_task = loop.run_in_executor(None, get_technical_data, ticker)
    insider_task = loop.run_in_executor(None, get_insider_activity, ticker)
    news_task = loop.run_in_executor(None, get_news_summary, ticker)
    
    tech, insider, news = await asyncio.gather(tech_task, insider_task, news_task)
    
    if not tech:
        await msg.edit(content=f"❌ ไม่พบข้อมูลหุ้น {ticker}")
        return

    # 2. ส่งให้ Gemini ตัดสิน
    await msg.edit(content=f"⚖️ **ผู้พิพากษา AI กำลังพิจารณาคำตัดสิน...**")
    verdict = await loop.run_in_executor(None, consult_the_judge, ticker, tech, insider, news)
    
    # 3. รายงานผล
    embed = discord.Embed(title=f"🏛️ คำพิพากษา: {ticker}", color=0xf1c40f)
    embed.add_field(name="ราคาปัจจุบัน", value=f"${tech['price']:.2f}", inline=True)
    embed.add_field(name="RSI", value=f"{tech['rsi']:.1f}", inline=True)
    embed.add_field(name="แนวโน้ม", value=tech['trend'], inline=True)
    embed.description = f"**👨‍⚖️ มุมมองจาก The Judge:**\n{verdict}"
    
    await ctx.send(embed=embed)
    await msg.delete()

if __name__ == "__main__":
    keep_alive()
    if DISCORD_TOKEN:
        bot.run(DISCORD_TOKEN)

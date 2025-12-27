# ==============================================================================
# 🤖 PROJECT: THE SOVEREIGN GUARDIAN (FIXED SYNTAX VERSION)
# ==============================================================================

import os
import json
import random
import discord
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

# --- 🌐 WEB SERVER (Keep Alive) ---
app = Flask('')
@app.route('/')
def home(): return "SYSTEM ONLINE: Sovereign Engine Running."
def run(): app.run(host='0.0.0.0', port=8080)
def keep_alive(): t = Thread(target=run); t.start()

# --- 🔑 CONFIGURATION ---
DISCORD_TOKEN = os.environ.get('DISCORD_TOKEN')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
PORTFOLIOS_FILE = 'user_portfolios.json'
TRADING_FILE = 'paper_trading.json'
HISTORY_FILE = 'prediction_history.json'
START_DATE = '2020-01-01'
PREDICTION_DAYS = 60

# ตั้งค่า Gemini
try:
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-pro')
except: pass

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

# ==============================================================================
# 🗄️ DATABASE MANAGER (แก้ Syntax แล้ว)
# ==============================================================================
def load_json(filename):
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_json(filename, data):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

# ==============================================================================
# 🎮 PAPER TRADING (ระบบเทรดจำลอง)
# ==============================================================================
INITIAL_BALANCE = 100000.0

def get_trader_profile(user_id):
    data = load_json(TRADING_FILE)
    if str(user_id) not in data:
        data[str(user_id)] = {"balance": INITIAL_BALANCE, "portfolio": {}, "history": []}
        save_json(TRADING_FILE, data)
    return data[str(user_id)]

def execute_trade(user_id, ticker, action, amount):
    data = load_json(TRADING_FILE)
    profile = data.get(str(user_id))
    if not profile: return False, "ต้องเปิดบัญชีก่อนครับ พิมพ์ `!register`"

    ticker = ticker.upper()
    try:
        current_price = yf.Ticker(ticker).history(period="1d")['Close'].iloc[-1]
    except: return False, "ไม่พบข้อมูลหุ้นตัวนี้"
    
    cost = current_price * amount

    if action == "BUY":
        if profile['balance'] >= cost:
            profile['balance'] -= cost
            profile['portfolio'][ticker] = profile['portfolio'].get(ticker, 0) + amount
            profile['history'].append(f"BUY {amount} {ticker} @ ${current_price:.2f}")
            save_json(TRADING_FILE, data)
            return True, f"✅ **ซื้อสำเร็จ!** {ticker} x {amount} หุ้น (ราคา ${current_price:.2f})"
        else: return False, "❌ เงินไม่พอครับ!"

    elif action == "SELL":
        current_holdings = profile['portfolio'].get(ticker, 0)
        if current_holdings >= amount:
            profile['balance'] += cost
            profile['portfolio'][ticker] -= amount
            if profile['portfolio'][ticker] == 0: del profile['portfolio'][ticker]
            profile['history'].append(f"SELL {amount} {ticker} @ ${current_price:.2f}")
            save_json(TRADING_FILE, data)
            return True, f"✅ **ขายสำเร็จ!** {ticker} x {amount} หุ้น (รับเงิน ${cost:.2f})"
        else: return False, "❌ มีหุ้นไม่พอขายครับ!"
    
    return False, "Error"

# ==============================================================================
# 🏥 FUNDAMENTAL & TECHNICAL ENGINE
# ==============================================================================
def check_fundamentals(ticker):
    """ตรวจสุขภาพการเงิน: พื้นฐานเปลี่ยนไหม?"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        pe = info.get('trailingPE', 0)
        margin = info.get('profitMargins', 0)
        growth = info.get('revenueGrowth', 0)
        
        status = "พื้นฐานแกร่ง 💪"
        score = 0
        
        if margin > 0: score += 1      # มีกำไร
        if growth > 0: score += 1      # เติบโต
        if 0 < pe < 60: score += 1     # ราคาไม่เวอร์
        
        if margin < 0: 
            status = "ขาดทุน (พื้นฐานแย่) 🩸"
            score = -1
        elif score == 0: 
            status = "อาการน่าเป็นห่วง 😷"
        
        return {"status": status, "score": score, "pe": pe, "margin": margin}
    except: return {"status": "ไม่ทราบ", "score": 0}

def analyze_market(ticker):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    try:
        df = yf.download(ticker, start=START_DATE, progress=False)
        if len(df) < 100: return None
        curr_price = df['Close'].iloc[-1].item()
        
        # Technicals (RSI)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        curr_rsi = (100 - (100 / (1 + (gain / loss)))).iloc[-1].item()
        
        # AI Forecast (LSTM)
        data = df[['Close']].values
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)
        X = np.array([scaled_data[-PREDICTION_DAYS:]])
        
        model_file = f'brain_{ticker}.keras'
        if not os.path.exists(model_file):
            ai_model = Sequential([LSTM(50, input_shape=(PREDICTION_DAYS, 1)), Dense(1)])
            ai_model.compile(optimizer='adam', loss='mse')
            ai_model.fit(X, np.array([scaled_data[-1]]), epochs=5, verbose=0)
            ai_model.save(model_file)
        else:
            ai_model = load_model(model_file)
            
        pred_price = scaler.inverse_transform(ai_model.predict(X, verbose=0))[0][0]
        
        # Fundamentals & Insider & News
        fund = check_fundamentals(ticker)
        
        stock = yf.Ticker(ticker)
        insider = stock.insider_transactions
        insider_txt = "ปกติ"
        if insider is not None and not insider.empty:
             if sum(1 for i, r in insider.head(5).iterrows() if "sale" in str(r.get('Transaction','')).lower()) >= 2:
                 insider_txt = "🚨 ผู้บริหารเทขาย"

        news_score = 0
        news_txt = "ไม่มีข่าว"
        if GEMINI_API_KEY:
            news = stock.news[:3]
            if news:
                headlines = "\n".join([f"- {n['title']}" for n in news])
                try:
                    prompt = f"Rate sentiment (-1.0 to 1.0) for these headlines. Return ONLY the number:\n{headlines}"
                    res = model.generate_content(prompt)
                    news_score = float(res.text.strip())
                    news_txt = headlines
                except: pass

        # --- FINAL SCORING ---
        ai_score = np.clip(((pred_price - curr_price)/curr_price)*10, -1, 1)
        rsi_score = 1 if curr_rsi < 30 else -1 if curr_rsi > 70 else 0
        fund_impact = (fund['score'] - 1) / 2
        
        final_score = (ai_score * 0.3) + (rsi_score * 0.2) + (news_score * 0.2) + (fund_impact * 0.3)
        
        if fund['score'] == -1: final_score = -1.0 
        if "เทขาย" in insider_txt: final_score -= 0.5

        signal = "HOLD ✋"
        color = 0x95a5a6
        if final_score > 0.35: signal = "BUY NOW 🟢"; color = 0x2ecc71
        elif final_score < -0.35: signal = "SELL NOW 🔴"; color = 0xe74c3c
        
        return {
            "price": curr_price, "signal": signal, "score": final_score, 
            "color": color, "fund": fund, "insider": insider_txt,
            "rsi": curr_rsi, "news": news_txt
        }
    except Exception as e: return None

def consult_mentor(ticker, data):
    if not GEMINI_API_KEY: return "No Gemini"
    prompt = f"""
    สอนเพื่อนมือใหม่เกี่ยวกับหุ้น {ticker} แบบเข้าใจง่ายๆ (เปรียบเทียบกับชีวิตจริง):
    - สุขภาพบริษัท: {data['fund']['status']}
    - สัญญาณกราฟ: {data['signal']} (คะแนน {data['score']:.2f})
    - ผู้บริหาร: {data['insider']}
    
    สรุปสั้นๆว่าควรทำยังไง (ซื้อ/ขาย/หนี)
    """
    try: return model.generate_content(prompt).text
    except: return "ครูไม่ว่างครับ"

# ==============================================================================
# 🥷 NINJA ALERT SYSTEM (STEALTH & REAL-TIME)
# ==============================================================================
@tasks.loop(seconds=1) 
async def ninja_alert_task():
    await bot.wait_until_ready()
    
    while not bot.is_closed():
        print("🥷 Ninja Scan Started...")
        portfolios = load_json(PORTFOLIOS_FILE)
        
        all_tickers = set()
        for tickers in portfolios.values():
            for t in tickers: all_tickers.add(t)
        
        if not all_tickers:
            await asyncio.sleep(60)
            continue

        for ticker in all_tickers:
            try:
                # สุ่มพัก 5-10 วินาที
                sleep_time = random.uniform(5, 10)
                await asyncio.sleep(sleep_time)
                
                loop = asyncio.get_running_loop()
                data = await loop.run_in_executor(None, analyze_market, ticker)
                
                if data:
                    if "NOW" in data['signal']:
                        print(f"🚨 FOUND SIGNAL: {ticker} -> {data['signal']}")
                        
                        embed = discord.Embed(title=f"🚨 ALERT: {ticker}", color=data['color'])
                        embed.add_field(name="สัญญาณ", value=f"**{data['signal']}**", inline=True)
                        embed.add_field(name="ราคา", value=f"${data['price']:.2f}", inline=True)
                        embed.add_field(name="พื้นฐาน", value=data['fund']['status'], inline=True)
                        embed.set_footer(text="Ninja Real-time Watch")
                        
                        for user_id, user_tickers in portfolios.items():
                            if ticker in user_tickers:
                                try:
                                    user = bot.get_user(int(user_id))
                                    if user: await user.send(f"🔥 **{ticker}** มาแล้วครับ!", embed=embed)
                                except: pass
            except: pass

        cooldown = random.uniform(30, 60)
        print(f"☕ Cooling down {cooldown:.0f}s...")
        await asyncio.sleep(cooldown)

# ==============================================================================
# 🎮 COMMANDS
# ==============================================================================
@bot.event
async def on_ready():
    print(f'🤖 Sovereign Bot Online: {bot.user}')
    if not ninja_alert_task.is_running():
        ninja_alert_task.start()

@bot.command()
async def register(ctx):
    """เปิดบัญชีเทรดจำลอง"""
    data = get_trader_profile(ctx.author.id)
    await ctx.send(f"🎉 เปิดบัญชีให้ **{ctx.author.name}** แล้ว! มีเงินเริ่มต้น **${data['balance']:,.2f}**")

@bot.command()
async def wallet(ctx):
    """เช็คกระเป๋าเงิน"""
    data = get_trader_profile(ctx.author.id)
    embed = discord.Embed(title=f"💼 พอร์ตจำลอง: {ctx.author.name}", color=0x3498db)
    embed.add_field(name="💵 เงินสด", value=f"${data['balance']:,.2f}", inline=False)
    
    if data['portfolio']:
        txt = ""
        total = data['balance']
        for t, amt in data['portfolio'].items():
            try:
                p = yf.Ticker(t).history(period="1d")['Close'].iloc[-1]
                val = p * amt
                total += val
                txt += f"- **{t}**: {amt} หุ้น (${val:,.2f})\n"
            except: pass
        embed.add_field(name="📈 หุ้นที่ถือ", value=txt, inline=False)
        embed.set_footer(text=f"มูลค่ารวมสุทธิ: ${total:,.2f}")
    
    await ctx.send(embed=embed)

@bot.command()
async def buy(ctx, ticker: str, amount: int):
    """ซื้อหุ้นจำลอง"""
    success, msg = execute_trade(ctx.author.id, ticker, "BUY", amount)
    await ctx.send(msg)

@bot.command()
async def sell(ctx, ticker: str, amount: int):
    """ขายหุ้นจำลอง"""
    success, msg = execute_trade(ctx.author.id, ticker, "SELL", amount)
    await ctx.send(msg)

@bot.command()
async def check(ctx, ticker: str):
    """วิเคราะห์หุ้นเต็มสูบ"""
    t = ticker.upper()
    msg = await ctx.send(f"🔍 **กำลังสแกน {t} (พื้นฐาน+กราฟ+ข่าว)...**")
    
    loop = asyncio.get_running_loop()
    data = await loop.run_in_executor(None, analyze_market, t)
    
    if not data: await msg.edit(content="❌ ไม่พบข้อมูล"); return
    
    advice = await loop.run_in_executor(None, consult_mentor, t, data)
    
    embed = discord.Embed(title=f"💎 Report: {t}", color=data['color'])
    embed.add_field(name="สัญญาณ", value=f"**{data['signal']}**", inline=True)
    embed.add_field(name="พื้นฐาน", value=data['fund']['status'], inline=True)
    embed.add_field(name="ผู้บริหาร", value=data['insider'], inline=True)
    embed.description = f"**👨‍🏫 ครูพี่เลี้ยง:**\n{advice}"
    
    await ctx.send(embed=embed)
    await msg.delete()

@bot.command()
async def add(ctx, *tickers):
    """เพิ่มหุ้นเข้า Watchlist"""
    uid=str(ctx.author.id); p=load_json(PORTFOLIOS_FILE); up=p.get(uid,[]); 
    added = [t.upper() for t in tickers if t.upper() not in up]
    up.extend(added); p[uid]=up; save_json(PORTFOLIOS_FILE,p); 
    await ctx.send(f"✅ เพิ่ม {', '.join(added)} เข้า Watchlist แล้ว (บอทนินจากำลังเริ่มเฝ้า...)")

@bot.command()
async def teach(ctx, term: str):
    """สอนศัพท์หุ้น"""
    if GEMINI_API_KEY:
        res = model.generate_content(f"Explain '{term}' simply/funnily for beginner (Thai).")
        await ctx.send(f"🎓 **{term}:**\n{res.text}")

if __name__ == "__main__":
    keep_alive()
    if DISCORD_TOKEN: bot.run(DISCORD_TOKEN)

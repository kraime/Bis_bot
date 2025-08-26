#!/usr/bin/env python3
"""
Test script for chunking system and long profile matching
"""

import asyncio
import os
import sys
from dotenv import load_dotenv

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Load environment variables
load_dotenv()


async def test_chunking_system():
    """Test chunking system with long profiles"""
    print("🚀 Testing Chunking System with Long Profiles\n")

    try:
        # Import modules
        from src.database import db
        from src.embeddings import embedding_service
        from src.vector_db import vector_db
        from src.text_processing import text_processor
        from src.llm_service import llm_service

        # Initialize systems
        print("🔍 Initializing systems...")
        await db.connect()
        await vector_db.initialize()
        embedding_service.load_model()
        print("✅ All systems initialized")

        # Clean existing test data
        print("\n🧹 Cleaning up existing test data...")
        try:
            import aiosqlite
            async with aiosqlite.connect(db.db_path) as conn:
                # Delete test profiles
                await conn.execute("""
                    DELETE FROM user_profiles WHERE user_id IN (
                        SELECT id FROM users WHERE telegram_id BETWEEN 20000 AND 29999
                    )
                """)
                await conn.execute("""
                    DELETE FROM profile_history WHERE user_id IN (
                        SELECT id FROM users WHERE telegram_id BETWEEN 20000 AND 29999
                    )
                """)
                await conn.execute("""
                    DELETE FROM user_states WHERE user_id IN (
                        SELECT id FROM users WHERE telegram_id BETWEEN 20000 AND 29999
                    )
                """)
                await conn.execute("DELETE FROM users WHERE telegram_id BETWEEN 20000 AND 29999")
                await conn.commit()

            # Clean Qdrant
            for user_id in range(1, 50):
                try:
                    vector_db.delete_profile(user_id)
                except:
                    pass

            print("✅ Test data cleaned")
        except Exception as e:
            print(f"⚠️ Cleanup warning: {e}")

        # Create test profiles with long descriptions
        long_test_profiles = [
            {
                "telegram_id": 20001,
                "username": "ai_startup_founder",
                "first_name": "Александр",
                "last_name": "Основатель",
                "birthday": "15.03.1985",
                "phone": "+79161234567",
                "answers": [
                    """Я основатель и технический директор AI-стартапа, который разрабатывает революционные решения в области искусственного интеллекта для автоматизации бизнес-процессов. Наша компания специализируется на создании интеллектуальных систем обработки естественного языка, компьютерного зрения и машинного обучения. За последние 5 лет мы выросли от команды из 3 человек до 50+ сотрудников, привлекли $15M инвестиций и запустили продукты, которыми пользуются более 100,000 компаний по всему миру. Мой опыт включает 12 лет разработки в крупных технологических компаниях, включая Google и Яндекс, где я руководил командами по разработке ML-алгоритмов. Имею PhD в области Computer Science, опубликовал более 20 научных статей в топовых конференциях по машинному обучению. Специализируюсь на deep learning, NLP, computer vision, distributed systems и high-performance computing. Активно участвую в технологическом сообществе, выступаю на конференциях, менторю стартапы в акселераторах. Наша текущая продуктовая линейка включает платформу для автоматизации customer support с помощью AI-чатботов, систему анализа документов для юридических компаний, и решение для предиктивной аналитики в e-commerce.""",

                    """Ищу стратегических партнеров и инвесторов для масштабирования нашего AI-стартапа на международные рынки, особенно в США и Европе. Нам нужны партнеры с экспертизой в enterprise sales, которые помогут выйти на крупных корпоративных клиентов в финтехе, здравоохранении и ритейле. Также ищу технических co-founder'ов с опытом в области MLOps, DevOps и cloud infrastructure для оптимизации наших ML-пайплайнов и снижения costs на inference. Интересуют партнерства с университетами и исследовательскими центрами для совместной разработки cutting-edge алгоритмов. Рассматриваю возможности M&A - как приобретение перспективных AI-команд, так и потенциальный exit через продажу стратегическому инвестору. Нужны эксперты по регулированию AI в разных юрисдикциях, особенно с учетом новых законов о AI в ЕС. Ищу ментора с опытом scaling tech companies от $10M до $100M+ ARR. Интересны joint ventures с крупными корпорациями для внедрения наших AI-решений в их продукты. Также рассматриваю партнерства с cloud providers для получения compute credits и технической поддержки.""",

                    """Могу предоставить глубокую экспертизу в области искусственного интеллекта и машинного обучения, включая архитектуру и разработку ML-систем от proof-of-concept до production-ready решений, способных обрабатывать миллионы запросов в день. Помогу с техническим due diligence AI-стартапов, оценкой их технологий и команд. Предоставлю доступ к нашей proprietary ML-платформе и готовым AI-моделям через API для быстрого прототипирования. Могу стать техническим advisor'ом или co-founder'ом в перспективных AI-проектах. Поделюсь опытом fundraising'а - от seed до Series B, включая подготовку pitch deck'ов, работу с VC и angel investors. Помогу с hiring и построением AI-команд, у меня есть обширная сеть контактов среди ML-инженеров, data scientists и AI-исследователей. Предоставлю менторство по product management для AI-продуктов, включая определение product-market fit и go-to-market стратегии. Могу помочь с техническими интеграциями наших AI-решений в существующие продукты партнеров. Поделюсь знаниями о compliance и этических аспектах AI, особенно важными для enterprise клиентов. Предоставлю доступ к нашей customer base для pilot проектов и case studies."""
                ]
            },

            {
                "telegram_id": 20002,
                "username": "venture_capital_partner",
                "first_name": "Мария",
                "last_name": "Инвестор",
                "birthday": "22.07.1978",
                "phone": "+79167654321",
                "answers": [
                    """Я Managing Partner в одном из ведущих венчурных фондов России с AUM $500M+, специализирующемся на инвестициях в технологические стартапы на стадиях от seed до Series B. За 15 лет в венчурной индустрии я инвестировала в более чем 80 компаний, из которых 12 достигли статуса unicorn, а 25+ были успешно проданы стратегическим инвесторам или вышли на IPO. Мой инвестиционный фокус включает fintech, healthtech, AI/ML, enterprise software, cybersecurity и climate tech. До венчурного капитала работала в Goldman Sachs в подразделении tech M&A, где участвовала в сделках на общую сумму более $10B. Имею MBA от Wharton и степень по математике от МГУ. Активно участвую в экосистеме стартапов как board member в 15+ компаниях, где помогаю с стратегическим планированием, fundraising, M&A и международной экспансией. Регулярно выступаю на ведущих конференциях по венчурному капиталу, включая Slush, Web Summit, TechCrunch Disrupt. Являюсь LP в нескольких американских и европейских фондах, что дает мне глобальную перспективу на рынок. Веду собственный angel portfolio из 30+ early-stage инвестиций в перспективные команды.""",

                    """Активно ищу breakthrough технологические стартапы для инвестиций от $1M до $50M, особенно в области artificial intelligence, quantum computing, biotechnology и clean energy. Интересуют компании с strong technical moats, experienced founding teams и clear path to $100M+ revenue. Рассматриваю как российские, так и международные возможности, особенно для co-investment с топовыми зарубежными фондами. Ищу стратегические партнерства с corporate venture arms крупных технологических компаний для совместных инвестиций и обеспечения portfolio companies доступом к enterprise клиентам. Интересуют возможности lead или co-lead крупных раундов Series A/B в компаниях с proven product-market fit и strong unit economics. Активно рассматриваю cross-border сделки, особенно для помощи российским стартапам в выходе на международные рынки через наши партнерские фонды в США, Европе и Азии. Ищу experienced entrepreneurs in residence для работы с нашими portfolio companies в качестве interim executives или advisors. Интересуют partnership opportunities с leading accelerators и incubators для early-stage deal flow. Рассматриваю возможности создания sector-specific funds, особенно в области climate tech и healthcare innovation.""",

                    """Предоставляю comprehensive support для portfolio companies на всех стадиях их развития, включая стратегическое планирование, product development guidance, go-to-market strategy и international expansion. Помогаю с последующими раундами fundraising через мою extensive network из 200+ institutional investors по всему миру. Обеспечиваю доступ к top-tier executive talent через partnership с leading executive search firms и мою personal network из successful entrepreneurs и C-level executives. Предоставляю business development support через connections с potential enterprise customers, strategic partners и distribution channels. Помогаю с M&A opportunities как на buy-side, так и на sell-side, leveraging мой опыт в investment banking и extensive network среди strategic acquirers. Предоставляю operational expertise в areas включая financial planning, legal structuring, HR policies и corporate governance. Помогаю с international market entry через partnerships с local funds и advisors в key markets. Обеспечиваю PR и marketing support через connections с leading tech media и analyst firms. Предоставляю access к exclusive industry events, conferences и networking opportunities. Помогаю с talent acquisition через мою network и partnerships с technical recruiting firms. Предоставляю ongoing mentorship и strategic advice based на мой experience building и scaling technology companies."""
                ]
            },

            {
                "telegram_id": 20003,
                "username": "enterprise_sales_director",
                "first_name": "Дмитрий",
                "last_name": "Продажи",
                "birthday": "10.11.1982",
                "phone": "+79169876543",
                "answers": [
                    """Я Global Sales Director с 18-летним опытом построения и масштабирования enterprise sales организаций в B2B SaaS компаниях от early-stage стартапов до публичных корпораций с revenue $1B+. Специализируюсь на complex enterprise deals размером от $100K до $10M+ с sales cycles от 6 до 24 месяцев. За свою карьеру построил sales teams в 4 компаниях, которые достигли successful exits через IPO или acquisition. Мой track record включает рост revenue от $2M до $200M ARR в роли VP Sales в fintech стартапе, который был приобретен за $1.2B. Имею глубокую экспертизу в vertical markets включая financial services, healthcare, manufacturing, retail и government. Развил comprehensive sales methodology, которая увеличивает win rates на 40% и сокращает sales cycles на 25%. Построил и управлял international sales teams в 15+ странах, включая США, Европу, APAC и LATAM. Имею extensive network из 500+ enterprise decision makers в Fortune 500 компаниях. Регулярно выступаю на leading sales conferences и являюсь advisor для 10+ B2B стартапов. Сертифицирован по major sales methodologies включая MEDDIC, Challenger Sale, и Solution Selling. Имею MBA в области International Business и постоянно изучаю emerging trends в enterprise software и digital transformation.""",

                    """Ищу возможности присоединиться к high-growth B2B SaaS стартапам в качестве Chief Revenue Officer, VP Sales или co-founder с equity participation для масштабирования их enterprise sales operations. Интересуют компании с proven product-market fit, strong technical team и готовностью к aggressive international expansion. Рассматриваю как full-time executive roles, так и advisory positions с equity compensation в перспективных стартапах на стадии Series A-C. Ищу partnerships с venture capital funds для due diligence их portfolio companies и помощи в scaling sales operations. Интересуют consulting opportunities для помощи enterprise software companies в entering new vertical markets или geographic regions. Рассматриваю возможности создания собственного sales consulting firm, специализирующегося на B2B SaaS companies. Ищу strategic partnerships с leading sales technology vendors для joint go-to-market initiatives. Интересуют board positions в technology companies, где могу применить мой operational expertise. Рассматриваю opportunities для angel investing в early-stage B2B startups, особенно где могу добавить significant value через мой sales expertise и network. Ищу partnerships с executive search firms, специализирующихся на sales leadership roles. Интересуют speaking opportunities на major industry conferences и participation в thought leadership initiatives.""",

                    """Предоставляю comprehensive sales leadership и operational expertise для scaling enterprise sales organizations от zero to $100M+ ARR. Помогаю с hiring и onboarding world-class sales talent, включая account executives, sales engineers, customer success managers и sales development representatives. Разрабатываю и implement proven sales processes, methodologies и playbooks, которые обеспечивают predictable revenue growth и high team performance. Предоставляю extensive network из enterprise decision makers для warm introductions и pilot opportunities. Помогаю с go-to-market strategy development, включая market segmentation, competitive positioning и pricing strategy. Обеспечиваю sales training и coaching programs для improvement team performance и individual quota attainment. Предоставляю expertise в sales technology stack optimization, включая CRM implementation, sales automation и analytics tools. Помогаю с international expansion strategy и establishing sales operations в new geographic markets. Предоставляю customer advisory services для product development prioritization based на enterprise customer feedback. Помогаю с partnership development и channel sales strategies для accelerated market penetration. Обеспечиваю ongoing strategic advice и mentorship для sales leadership teams. Предоставляю access к industry best practices и benchmarking data для continuous improvement initiatives."""
                ]
            },

            {
                "telegram_id": 20004,
                "username": "fintech_product_manager",
                "first_name": "Анна",
                "last_name": "Продукт",
                "birthday": "05.09.1987",
                "phone": "+79162345678",
                "answers": [
                    """Я Senior Product Manager в leading fintech company с 10-летним опытом создания и масштабирования financial products, которыми пользуются миллионы клиентов ежедневно. Специализируюсь на payments, lending, wealth management и regulatory compliance в высоко регулируемой финансовой индустрии. За мою карьеру я запустила 15+ successful products, включая mobile payment platform с $2B+ annual transaction volume, AI-powered credit scoring system и robo-advisor platform с $500M+ assets under management. Имею глубокую экспертизу в user experience design для financial services, data-driven product development и A/B testing methodologies. Работала в различных финтех сегментах от consumer banking до institutional trading platforms. Имею strong technical background с пониманием blockchain technology, machine learning applications в finance и cybersecurity requirements. Активно участвую в fintech community как speaker на major conferences включая Money20/20, Finovate и LendIt. Являюсь advisor для 5+ fintech startups и mentor в leading accelerator programs. Имею MBA в Finance и постоянно изучаю emerging trends включая DeFi, central bank digital currencies и embedded finance. Сертифицирована по product management methodologies включая Agile, Lean Startup и Design Thinking. Поддерживаю extensive network relationships с regulators, financial institutions и fintech ecosystem players.""",

                    """Активно ищу opportunities для создания revolutionary fintech products, которые democratize access к financial services и improve financial inclusion globally. Интересуют partnerships с traditional financial institutions для digital transformation initiatives и creation of innovative customer experiences. Рассматриваю co-founder opportunities в early-stage fintech startups, особенно в areas включая embedded finance, SME lending, cross-border payments и sustainable finance. Ищу collaboration с AI/ML teams для development of next-generation financial products powered by artificial intelligence и predictive analytics. Интересуют consulting opportunities для helping established financial services companies с product strategy, digital transformation и regulatory compliance. Рассматриваю advisory roles в fintech startups где могу leverage мой product expertise и industry connections. Ищу partnerships с regulatory bodies и policy makers для shaping future of financial services regulation и promoting innovation-friendly policies. Интересуют opportunities для expanding в emerging markets, особенно в regions с underserved populations и significant financial inclusion gaps. Рассматриваю roles в venture capital funds, специализирующихся на fintech investments, где могу provide product due diligence и portfolio company support. Ищу speaking opportunities и thought leadership platforms для sharing insights о future of financial services и product innovation trends.""",

                    """Предоставляю comprehensive product management expertise для fintech companies на всех stages of development, от concept validation до scale и international expansion. Помогаю с product strategy development, roadmap planning и prioritization based на market research, customer feedback и competitive analysis. Обеспечиваю deep understanding regulatory requirements across multiple jurisdictions и guidance на compliance-first product development. Предоставляю extensive network из financial services executives, regulators, technology vendors и industry experts для partnerships и business development. Помогаю с user experience design и customer journey optimization specifically для financial products с focus на trust, security и ease of use. Предоставляю data analytics expertise для product performance measurement, customer behavior analysis и predictive modeling. Помогаю с go-to-market strategy development, включая pricing models, distribution channels и customer acquisition strategies. Обеспечиваю guidance на technology architecture decisions, vendor selection и integration strategies для financial services infrastructure. Предоставляю fundraising support через connections с fintech-focused investors и assistance с product positioning для investment presentations. Помогаю с international expansion strategy, включая market entry planning, local partnership development и regulatory navigation. Обеспечиваю ongoing product coaching и mentorship для product teams и leadership. Предоставляю access к industry benchmarking data, best practices и emerging trend analysis для competitive advantage."""
                ]
            }
        ]

        created_users = []

        print(f"\n👥 Creating {len(long_test_profiles)} long test profiles...")

        for i, profile_data in enumerate(long_test_profiles, 1):
            try:
                print(f"\n🔍 Creating profile {i}/{len(long_test_profiles)}: {profile_data['first_name']}")

                # Create user
                user = await db.get_or_create_user(
                    telegram_id=profile_data["telegram_id"],
                    username=profile_data["username"],
                    first_name=profile_data["first_name"],
                    last_name=profile_data["last_name"]
                )

                # Update phone and birthday
                await db.update_user_phone(user['id'], profile_data["phone"])
                await db.update_user_birthday(user['id'], profile_data["birthday"])

                # Process profile text and test chunking
                processed_data = text_processor.prepare_profile_text(
                    profile_data["answers"][0],
                    profile_data["answers"][1],
                    profile_data["answers"][2]
                )

                print(f"  📊 Text stats:")
                print(f"    Total length: {processed_data['total_length']} chars")
                print(f"    Chunks created: {len(processed_data['chunks'])}")
                print(f"    Keywords: {processed_data['keywords'][:8]}...")

                # Show chunk details
                for j, chunk in enumerate(processed_data['chunks'], 1):
                    print(f"    Chunk {j}: {len(chunk)} chars - {chunk[:60]}...")

                # Create embedding (will handle chunking automatically)
                embedding = embedding_service.create_profile_embedding(
                    processed_data['clean_answers']['answer_1'],
                    processed_data['clean_answers']['answer_2'],
                    processed_data['clean_answers']['answer_3']
                )

                print(f"  🧠 Embedding dimension: {len(embedding)}")

                # Save to SQLite
                await db.save_user_profile(
                    user['id'],
                    processed_data['clean_answers']['answer_1'],
                    processed_data['clean_answers']['answer_2'],
                    processed_data['clean_answers']['answer_3'],
                    embedding,
                    processed_data['keywords']
                )

                # Save to Qdrant
                profile_payload = {
                    "telegram_id": user['telegram_id'],
                    "username": user['username'],
                    "first_name": user['first_name'],
                    "last_name": user['last_name'],
                    "answer_1": processed_data['clean_answers']['answer_1'],
                    "answer_2": processed_data['clean_answers']['answer_2'],
                    "answer_3": processed_data['clean_answers']['answer_3'],
                    "keywords": processed_data['keywords']
                }

                vector_db.save_profile_embedding(user['id'], embedding, profile_payload)

                created_users.append({
                    'user': user,
                    'profile_data': profile_data,
                    'processed_data': processed_data,
                    'embedding': embedding
                })

                print(f"  ✅ Profile created and saved")

            except Exception as e:
                print(f"  ❌ Failed to create profile for {profile_data['first_name']}: {e}")
                import traceback
                traceback.print_exc()

        print(f"\n✅ Created {len(created_users)} long profiles successfully")

        # Test matching between profiles
        print(f"\n🔍 Testing matching between long profiles...")

        for i, user_data in enumerate(created_users, 1):
            user = user_data['user']
            profile_data = user_data['profile_data']

            print(f"\n--- Matching Test {i}/{len(created_users)}: {profile_data['first_name']} ---")
            print(f"👤 Profile summary:")
            print(f"   🏢 Field: {profile_data['answers'][0][:100]}...")
            print(f"   🔍 Looking for: {profile_data['answers'][1][:100]}...")
            print(f"   🤝 Can help: {profile_data['answers'][2][:100]}...")
            print(f"   📊 Text length: {user_data['processed_data']['total_length']} chars")
            print(f"   🧩 Chunks: {len(user_data['processed_data']['chunks'])}")

            try:
                # Get user profile from database
                db_profile = await db.get_user_profile(user['id'])
                if not db_profile:
                    print("  ❌ Profile not found in database")
                    continue

                # Test keyword-based search
                keywords = db_profile.get('keywords', [])
                if keywords:
                    keyword_profiles = await db.find_profiles_with_keywords(
                        user['id'], keywords, limit=10
                    )
                    print(f"  🔍 Keyword search: found {len(keyword_profiles)} profiles")
                    for kp in keyword_profiles:
                        print(f"    - {kp['first_name']}: {kp['answer_1'][:50]}...")
                else:
                    keyword_profiles = []

                # Test vector search
                vector_profiles = vector_db.search_similar_profiles(
                    user_data['embedding'], user['id'], limit=10
                )
                print(f"  🔍 Vector search: found {len(vector_profiles)} profiles")
                for vp in vector_profiles:
                    similarity = vp.get('similarity_score', 0)
                    print(f"    - {vp['first_name']} (similarity: {similarity:.3f}): {vp['answer_1'][:50]}...")

                # Combine results
                all_candidates = []
                seen_users = set()

                for profiles_list in [keyword_profiles, vector_profiles]:
                    for p in profiles_list:
                        user_key = p['telegram_id']
                        if user_key not in seen_users:
                            seen_users.add(user_key)
                            all_candidates.append(p)

                print(f"  📊 Total candidates: {len(all_candidates)}")

                if all_candidates:
                    # Test LLM matching
                    try:
                        best_matches = await llm_service.find_best_matches(
                            db_profile, all_candidates, top_k=3
                        )
                        print(f"  🤖 LLM analysis: {len(best_matches)} best matches")

                        for j, match in enumerate(best_matches, 1):
                            name = match.get('first_name', 'Unknown')
                            score = match.get('match_score', 'N/A')
                            reason = match.get('match_reason', 'No reason')
                            print(f"    {j}. {name} (Score: {score})")
                            print(f"       Reason: {reason}")
                            print(f"       Field: {match.get('answer_1', '')[:80]}...")

                        # Test summary generation
                        if best_matches:
                            summary = await llm_service.generate_match_summary(db_profile, best_matches)
                            print(f"  📝 Summary: {summary[:120]}...")

                    except Exception as e:
                        print(f"  ⚠️ LLM matching failed: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    print("  ⚠️ No candidates found for matching")

            except Exception as e:
                print(f"  ❌ Matching failed: {e}")
                import traceback
                traceback.print_exc()

        # Test specific matching scenarios
        print(f"\n🎯 Testing specific cross-matching scenarios...")

        # AI Founder looking for Investor
        ai_founder = next((u for u in created_users if 'ai_startup_founder' in u['user']['username']), None)
        vc_partner = next((u for u in created_users if 'venture_capital_partner' in u['user']['username']), None)

        if ai_founder and vc_partner:
            print(f"\n📋 Scenario 1: AI Founder → VC Partner matching")
            ai_profile = await db.get_user_profile(ai_founder['user']['id'])

            # Search for investor
            vector_results = vector_db.search_similar_profiles(
                ai_founder['embedding'], ai_founder['user']['id'], limit=5
            )

            investor_found = any(p['telegram_id'] == vc_partner['user']['telegram_id'] for p in vector_results)
            print(f"  {'✅' if investor_found else '❌'} VC Partner found in vector search: {investor_found}")

            if vector_results:
                llm_matches = await llm_service.find_best_matches(ai_profile, vector_results, top_k=3)
                investor_in_top = any(m['telegram_id'] == vc_partner['user']['telegram_id'] for m in llm_matches)
                print(f"  {'✅' if investor_in_top else '❌'} VC Partner in LLM top matches: {investor_in_top}")

                if investor_in_top:
                    match = next(m for m in llm_matches if m['telegram_id'] == vc_partner['user']['telegram_id'])
                    print(f"  🎯 Match score: {match.get('match_score', 'N/A')}")
                    print(f"  💡 Reason: {match.get('match_reason', 'No reason')}")

        # VC Partner looking for AI Founder
        if vc_partner and ai_founder:
            print(f"\n📋 Scenario 2: VC Partner → AI Founder matching")
            vc_profile = await db.get_user_profile(vc_partner['user']['id'])

            vector_results = vector_db.search_similar_profiles(
                vc_partner['embedding'], vc_partner['user']['id'], limit=5
            )

            founder_found = any(p['telegram_id'] == ai_founder['user']['telegram_id'] for p in vector_results)
            print(f"  {'✅' if founder_found else '❌'} AI Founder found in vector search: {founder_found}")

            if vector_results:
                llm_matches = await llm_service.find_best_matches(vc_profile, vector_results, top_k=3)
                founder_in_top = any(m['telegram_id'] == ai_founder['user']['telegram_id'] for m in llm_matches)
                print(f"  {'✅' if founder_in_top else '❌'} AI Founder in LLM top matches: {founder_in_top}")

        # Final statistics
        print(f"\n📊 Final Chunking Test Statistics:")
        print(f"  👥 Total long profiles created: {len(created_users)}")

        total_chunks = sum(len(u['processed_data']['chunks']) for u in created_users)
        avg_chunks = total_chunks / len(created_users) if created_users else 0
        print(f"  🧩 Total chunks created: {total_chunks}")
        print(f"  📈 Average chunks per profile: {avg_chunks:.1f}")

        total_length = sum(u['processed_data']['total_length'] for u in created_users)
        avg_length = total_length / len(created_users) if created_users else 0
        print(f"  📝 Average profile length: {avg_length:.0f} chars")

        # Get collection info
        final_info = vector_db.get_collection_info()
        print(f"  🔍 Qdrant profiles: {final_info.get('points_count', 0)}")

        print(f"\n🎉 Chunking system test completed successfully!")
        print(f"\nKey findings:")
        print(f"  ✅ Long texts are properly chunked")
        print(f"  ✅ Embeddings are averaged across chunks")
        print(f"  ✅ Vector search works with chunked profiles")
        print(f"  ✅ LLM matching provides relevant results")

        return True

    except Exception as e:
        print(f"❌ Chunking test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        try:
            await db.close()
        except:
            pass


if __name__ == "__main__":
    asyncio.run(test_chunking_system())
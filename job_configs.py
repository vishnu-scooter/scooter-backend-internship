# job_configs.py

# Define job IDs
JOB_ID_TESTZEUS_FOUNDING_BDR = "685ce18f2e09e59b2e9afcc8"
JOB_ID_ZENOTI_SENIOR_ACCOUNT_EXECUTIVE = "686e6a8f967f160e1c447a5d" 
JOB_ID_SAI_MARKETING_SALES_MANAGER = "68679d053a72380f84a62458" 
JOB_ID_ZENOTI_SALES_DEVELOPMENT_REPRESENTATIVE="686e6a547bbea9491c14e165"
JOB_ID_CLICKPOST="68905b9d5925cca675f43e00"
JOB_ID_ASTER_HEALTHCARE="68905c243382d9d111687565"
JOB_ID_GERMIN8="68c28debb42423989287fe57"
JOB_ID_DEMO= "68302465e9d6f5c2f783ca92"

# TestZeus Founding BDR Configuration
TESTZEUS_FOUNDING_BDR_CONFIG = {
    "job_id": JOB_ID_TESTZEUS_FOUNDING_BDR,
    "job_role": "TestZeus Founding BDR",
    "interview_questions": [
        {
        "question_number": 1,
        "question": "You’ve been reaching out to VP Engineering and CTOs at SaaS companies for three weeks with targeted messages but have received only one lukewarm reply. What would you do next to improve your results?",
        "evaluation_type": "Q1_OutboundAdaptation"
    },
    {
        "question_number": 2,
        "question": "Can you walk me through what your next outreach attempt would look like in this situation?",
        "evaluation_type": "Q1_OutboundAdaptation"
    },
    {
        "question_number": 3,
        "question": "You prepared for a prospect call, but your manager says you missed key buyer signals. You have 10 minutes before your next call. What would you do differently?",
        "evaluation_type": "Q2_BuyerSignalAdjustment"
    },
    {
        "question_number": 4,
        "question": "What specific signals would you look for or prepare to catch in your next call?",
        "evaluation_type": "Q2_BuyerSignalAdjustment"
    },
    {
        "question_number": 5,
        "question": "A CTO at a mid-sized SaaS company asks, ‘Why do we need something like TestZeus now?’ How would you explain it simply and clearly?",
        "evaluation_type": "Q3_ValueArticulation"
    },
    {
        "question_number": 6,
        "question": "How would you explain it in one or two sentences during a cold call?",
        "evaluation_type": "Q3_ValueArticulation"
    },
    {
        "question_number": 7,
        "question": "It’s Friday afternoon. You see a few warm prospects you have not followed up with yet. How would you decide what to do before ending your week?",
        "evaluation_type": "Q4_Prioritization"
    },
    {
        "question_number": 8,
        "question": "What usually helps you decide what’s worth following up on immediately vs. next week?",
        "evaluation_type": "Q4_Prioritization"
    },
    
    ],
    "evaluation_rubric": {
        "Q1_OutboundAdaptation": {
            "prompt_instructions": "Evaluate based on experimentation mindset, thoughtful sequencing, and creativity in outreach.",
            "scoring_logic": {
                "base_scoring": [
                    {"threshold": 0, "score": 1, "criteria": "No new ideas or vague plan"},
                    {"threshold": 1, "score": 2, "criteria": "One change (e.g., message tweak)"},
                    {"threshold": 2, "score": 3, "criteria": "Two tactics (e.g., channel switch, new persona)"},
                    {"threshold": 3, "score": 4, "criteria": "Three or more well-reasoned tactics"},
                    {"threshold": 4, "score": 5, "criteria": "Creative, multi-touch, hypothesis-driven strategy"}
                ],
                "keywords": ["experiment", "A/B", "test", "channel", "personalize", "sequence", "subject line", "reply rate"]
            }
        },
        "Q2_BuyerSignalAdjustment": {
            "prompt_instructions": "Evaluate based on recognition of specific buyer signals and agility in approach.",
            "scoring_logic": {
                "base_scoring": [
                    {"threshold": 0, "score": 1, "criteria": "No clear improvement plan"},
                    {"threshold": 1, "score": 2, "criteria": "Mentions vague signals"},
                    {"threshold": 2, "score": 3, "criteria": "Identifies 1-2 clear buyer cues"},
                    {"threshold": 3, "score": 4, "criteria": "Details a plan with 2-3 specific signal types"},
                    {"threshold": 4, "score": 5, "criteria": "Detailed improvement plan + signal examples + prep strategy"}
                ],
                "keywords": ["budget", "timeline", "decision maker", "pain point", "urgency", "buying stage"]
            }
        },
        "Q3_ValueArticulation": {
            "prompt_instructions": "Evaluate based on clarity, relevance to buyer persona, and TestZeus value connection.",
            "scoring_logic": {
                "clarity_levels": [
                    {"score": 1, "criteria": "Confusing or vague"},
                    {"score": 2, "criteria": "Some clarity but generic"},
                    {"score": 3, "criteria": "Clear, but limited to features"},
                    {"score": 4, "criteria": "Clear + ties to CTO priorities"},
                    {"score": 5, "criteria": "Clear, concise, and compelling with urgency"}
                ],
                "keywords": ["test coverage", "developer productivity", "AI", "open-source", "bug prevention", "speed", "release"]
            }
        },
        "Q4_Prioritization": {
            "prompt_instructions": "Evaluate based on logical prioritization and urgency signals.",
            "scoring_logic": {
                "base_scoring": [
                    {"threshold": 0, "score": 1, "criteria": "No plan or says 'log off'"},
                    {"threshold": 1, "score": 2, "criteria": "Follows gut or intuition only"},
                    {"threshold": 2, "score": 3, "criteria": "Mentions 1-2 prioritization criteria"},
                    {"threshold": 3, "score": 4, "criteria": "Evaluates based on prospect signals + opportunity stage"},
                    {"threshold": 4, "score": 5, "criteria": "Has a clear prioritization framework and action plan"}
                ],
                "keywords": ["last engagement", "deal stage", "urgency", "buying intent", "calendar block", "CRM notes"]
            }
        },
        "q_final":{
            "prompt_instructions": "Evaluate based on specificity in self-reflection and a growth mindset.",
            "scoring_logic": {
                "specificity_detection": [
                    {"type": "generic", "score": 1, "criteria": "All generic statements"},
                    {"type": "some_details", "score": 2, "criteria": "Some specific details"},
                    {"type": "1-2_examples", "score": 3, "criteria": "1-2 specific examples"},
                    {"type": "2-3_examples", "score": 4, "criteria": "2-3 specific examples"},
                    {"type": "multiple_detailed_examples", "score": 5, "criteria": "Multiple detailed examples"}
                ],
                "specificity_indicators": ["numbers", "company names", "specific situations", "metrics", "timeframes", "dollar amounts", "percentages"],
                "bonuses": [
                    {"type": "company_research", "score": 1, "keywords": ["Zenoti", "wellness industry", "competitors"]},
                    {"type": "growth_language", "score": 1, "keywords": ["learn", "improve", "develop", "coach", "feedback"]}
                ],
                "cap": 5,
                "notes": "Assesses the depth of self-reflection and openness to growth."
            }
        },
    },
    "trait_rubric": {
        "Grit": {
            "prompt_instructions": "Evaluated from Q1 + Q4. Look for persistence and action under pressure.",
            "criteria": [
                {"score": 5, "description": "Creative tactics + urgency + finishes strong"},
                {"score": 4, "description": "Multiple tactics + solid follow-through"},
                {"score": 3, "description": "Basic persistence and intent"},
                {"score": 2, "description": "Mild effort, limited urgency"},
                {"score": 1, "description": "Gives up or disengaged"}
            ],
            "source_question": [1, 4]
        },
        "Adaptability": {
            "prompt_instructions": "Evaluated from Q2. Look for reflection, signal interpretation, and fast adjustment.",
            "criteria": [
                {"score": 5, "description": "Rapid learning + clear buyer signal plan"},
                {"score": 4, "description": "Learns and adapts clearly"},
                {"score": 3, "description": "Some self-awareness"},
                {"score": 2, "description": "Struggles with adjustment"},
                {"score": 1, "description": "Defensive or unaware"}
            ],
            "source_question": 2
        },
        "Coachability": {
            "prompt_instructions": "Evaluated from Q5. Look for humility, learning mindset, and openness.",
            "criteria": [
                {"score": 5, "description": "Mentions learning, feedback, and growth directly"},
                {"score": 4, "description": "Shows willingness to adapt"},
                {"score": 3, "description": "Neutral or general positivity"},
                {"score": 2, "description": "Mild resistance or vague"},
                {"score": 1, "description": "Closed off or arrogant"}
            ],
            "source_question": 5
        }
    },
    "overall_decision_thresholds": [
        {"score_range": [40, 50], "decision": "Proceed", "action": "Move to next round"},
        {"score_range": [30, 39], "decision": "Maybe", "action": "Human review recommended"},
        {"score_range": [20, 29], "decision": "Weak", "action": "Likely reject, but flag for review"},
        {"score_range": [0, 19], "decision": "Do not proceed", "action": "Auto-reject"}
    ]
}
# Zenoti Senior Account Executive Configuration
ZENOTI_SENIOR_ACCOUNT_EXECUTIVE_CONFIG = {
    "job_id": JOB_ID_ZENOTI_SENIOR_ACCOUNT_EXECUTIVE,
    "job_role": "Zenoti Senior Account Executive",
    "interview_questions": [
        {
            "question_number": 1,
            "question": "You're talking to a spa chain owner. They say 'Our booking system is a mess and we can't track customers properly.' What questions would you ask next?",
            "evaluation_type": "Q1_DiscoverySkills"
        },
        {
            "question_number": 2,
            "question": "Your contact loves your solution but says 'My boss doesn't take vendor calls.' How do you get to the decision maker?",
            "evaluation_type": "Q2_StakeholderNavigation"
        },
        {
            "question_number": 3,
            "question": "A prospect says 'Your competitor is 25% cheaper. Can you match their price?' How do you respond?",
            "evaluation_type": "Q3_PriceObjections"
        },
        {
            "question_number": 4,
            "question": "Your deal is stuck. Your champion isn't responding and you're behind quota. What do you do next week?",
            "evaluation_type": "Q4_DealManagement"
        },
        {
            "question_number": 5,
            "question": "Someone asks 'What does Zenoti do?' How would you explain it briefly?",
            "evaluation_type": "Q5_ValueCommunication"
        }
    ],
    "evaluation_rubric": {
        "Q1_DiscoverySkills": {
            "prompt_instructions": "Evaluate based on number of questions asked and presence of business impact/stakeholder words.",
            "scoring_logic": {
                "base_scoring": [
                    {"threshold": 0, "score": 1, "criteria": "0 questions"},
                    {"threshold": 1, "score": 2, "criteria": "1-2 questions"},
                    {"threshold": 3, "score": 3, "criteria": "3-4 questions"},
                    {"threshold": 5, "score": 4, "criteria": "5-6 questions"},
                    {"threshold": 7, "score": 5, "criteria": "7+ questions"}
                ],
                "bonuses": [
                    {"keywords": ["revenue", "cost", "time", "efficiency", "growth", "money", "budget"], "score": 1, "type": "business_impact"},
                    {"keywords": ["who", "team", "decision", "manager", "owner"], "score": 1, "type": "stakeholder"}
                ],
                "cap": 5,
                "notes": "Counts questions and checks for specific keywords related to business impact and stakeholders."
            }
        },
        "Q2_StakeholderNavigation": {
            "prompt_instructions": "Evaluate based on distinct strategies mentioned for reaching decision-makers.",
            "scoring_logic": {
                "base_scoring": [
                    {"threshold": 0, "score": 1, "criteria": "0-1 strategies"},
                    {"threshold": 2, "score": 2, "criteria": "2 strategies"},
                    {"threshold": 3, "score": 3, "criteria": "3 strategies"},
                    {"threshold": 4, "score": 4, "criteria": "4 strategies"},
                    {"threshold": 5, "score": 5, "criteria": "5+ strategies"}
                ],
                "keywords": ["LinkedIn outreach", "asking for introduction", "scheduling meeting", "sending value content", "researching executive", "finding mutual connections", "attending events", "calling directly"],
                "penalties": [
                    {"keywords": ["force", "demand", "go around", "bypass", "ignore"], "score": -2, "min_score": 1}
                ],
                "notes": "Identifies distinct strategies and penalizes aggressive language."
            }
        },
        "Q3_PriceObjections": {
            "prompt_instructions": "Evaluate based on the sequence of response to price objections, focusing on value communication.",
            "scoring_logic": {
                "response_sequences": [
                    {"sequence": ["discount", "price"], "score": 1, "criteria": "Immediately mentions discount/matching price"},
                    {"sequence": ["price", "value"], "score": 2, "criteria": "Mentions price then talks about value"},
                    {"sequence": ["value", "price"], "score": 3, "criteria": "Mentions value then addresses price"},
                    {"sequence": ["questions", "value"], "score": 4, "criteria": "Asks questions then discusses value"},
                    {"sequence": ["questions", "value", "alternatives"], "score": 5, "criteria": "Asks questions + discusses value + offers alternatives"}
                ],
                "value_keywords": ["ROI", "return", "benefit", "savings", "efficiency", "productivity", "outcome", "investment", "worth"],
                "question_indicators": ["let me understand", "can you tell me", "what would", "how much", "why is"],
                "notes": "Analyzes the flow of the answer in handling price objections."
            }
        },
        "Q4_DealManagement": {
            "prompt_instructions": "Evaluate based on specific next actions mentioned for a stuck deal.",
            "scoring_logic": {
                "base_scoring": [
                    {"threshold": 0, "score": 1, "criteria": "0-1 actions"},
                    {"threshold": 2, "score": 2, "criteria": "2 actions"},
                    {"threshold": 3, "score": 3, "criteria": "3 actions"},
                    {"threshold": 4, "score": 4, "criteria": "4 actions"},
                    {"threshold": 5, "score": 5, "criteria": "5+ actions"}
                ],
                "action_examples": ["call", "email", "LinkedIn message", "research company news", "contact other stakeholders", "schedule meeting", "send follow-up content", "try different times/days"],
                "bonuses": [
                    {"type": "multiple_contact_methods", "score": 1, "note": "email AND phone AND LinkedIn"},
                    {"type": "timeline_schedule", "score": 1, "keywords": ["specific days", "deadlines"]}
                ],
                "cap": 5,
                "penalties": [
                    {"keywords": ["their fault", "unfair", "impossible", "won't work"], "score": -2, "min_score": 1}
                ],
                "notes": "Focuses on proactive, actionable steps with a positive mindset."
            }
        },
        "Q5_ValueCommunication": {
            "prompt_instructions": "Evaluate based on the explanation of Zenoti, considering timing and key content points.",
            "scoring_logic": {
                "timing": [
                    {"range": [0, 14], "score": 1, "criteria": "Under 15 seconds"},
                    {"range": [15, 30], "score": 2, "criteria": "15-30 seconds"},
                    {"range": [31, 60], "score": 5, "criteria": "30-60 seconds"},
                    {"range": [61, 90], "score": 4, "criteria": "60-90 seconds"},
                    {"range": [91, 1000], "score": 3, "criteria": "90+ seconds"}
                ],
                "content_requirements": [
                    {"type": "platform/integration", "keywords": ["all-in-one", "integrated", "single platform", "everything in one place"]},
                    {"type": "business_benefit", "keywords": ["efficiency", "growth", "save time", "increase revenue", "streamline"]},
                    {"type": "industry", "keywords": ["salon", "spa", "wellness", "beauty", "appointments", "bookings"]}
                ],
                "min_content_requirements_for_full_points": 2,
                "cap_if_missing_content": 3,
                "notes": "Balances conciseness with comprehensive explanation."
            }
        },
        "q_final":{
            "prompt_instructions": "Evaluate based on specificity in self-reflection and a growth mindset.",
            "scoring_logic": {
                "specificity_detection": [
                    {"type": "generic", "score": 1, "criteria": "All generic statements"},
                    {"type": "some_details", "score": 2, "criteria": "Some specific details"},
                    {"type": "1-2_examples", "score": 3, "criteria": "1-2 specific examples"},
                    {"type": "2-3_examples", "score": 4, "criteria": "2-3 specific examples"},
                    {"type": "multiple_detailed_examples", "score": 5, "criteria": "Multiple detailed examples"}
                ],
                "specificity_indicators": ["numbers", "company names", "specific situations", "metrics", "timeframes", "dollar amounts", "percentages"],
                "bonuses": [
                    {"type": "company_research", "score": 1, "keywords": ["Zenoti", "wellness industry", "competitors"]},
                    {"type": "growth_language", "score": 1, "keywords": ["learn", "improve", "develop", "coach", "feedback"]}
                ],
                "cap": 5,
                "notes": "Assesses the depth of self-reflection and openness to growth."
            }
        },
        
    },
    "trait_rubric": {
        "Resilience": {
            "prompt_instructions": "Evaluated from Q4. Look for action-focused, systematic approach without blame.",
            "criteria": [
                {"score": 5, "description": "No blame language + action-focused + systematic approach"},
                {"score": 4, "description": "Mostly positive + some clear actions"},
                {"score": 3, "description": "Mixed tone but has plan"},
                {"score": 2, "description": "Some negativity but still tries"},
                {"score": 1, "description": "Blame language or gives up"}
            ],
            "source_question": 4,
            "keywords_positive": ["action-focused", "systematic approach"],
            "keywords_negative": ["blame", "fault", "unfair", "impossible", "won't work"]
        },
        "Drive": {
            "prompt_instructions": "Evaluated from Q3 + Q4. Look for fighting for value, urgency, and multiple tactics.",
            "criteria": [
                {"score": 5, "description": "Fights for value + shows urgency + multiple tactics"},
                {"score": 4, "description": "Defends value + some urgency"},
                {"score": 3, "description": "Basic value defense + standard effort"},
                {"score": 2, "description": "Weak value defense + low energy"},
                {"score": 1, "description": "Quick to discount + passive approach"}
            ],
            "source_question": [3, 4],
            "keywords_positive": ["fights for value", "urgency", "multiple tactics"],
            "keywords_negative": ["quick to discount", "passive approach"]
        },
        "Coachability": {
            "prompt_instructions": "Evaluated from Q6. Look for self-awareness, growth mindset, and admitting gaps.",
            "criteria": [
                {"score": 5, "description": "High self-awareness + growth mindset + admits gaps"},
                {"score": 4, "description": "Good self-awareness + learning orientation"},
                {"score": 3, "description": "Some self-awareness + open to feedback"},
                {"score": 2, "description": "Limited self-awareness + defensive"},
                {"score": 1, "description": "No self-awareness + know-it-all attitude"}
            ],
            "source_question": 6,
            "keywords_positive": ["self-awareness", "growth mindset", "admits gaps", "learn", "improve", "develop", "coach", "feedback"],
            "keywords_negative": ["defensive", "know-it-all attitude"]
        }
    },
    "overall_decision_thresholds": [
        {"score_range": [35, 45], "decision": "Strong Hire", "action": "Move to next round"},
        {"score_range": [28, 34], "decision": "Maybe", "action": "Human review recommended"},
        {"score_range": [20, 27], "decision": "Weak", "action": "Likely reject, but flag for review"},
        {"score_range": [0, 19], "decision": "Reject", "action": "Auto-reject"}
    ]
}

# Sai Marketing Sales Manager Configuration
SAI_MARKETING_SALES_MANAGER_CONFIG = {
    "job_id": JOB_ID_SAI_MARKETING_SALES_MANAGER,
    "job_role": "Sai Marketing Sales Manager",
    "interview_questions": [
        {
            "question_number": 1,
            "question": "You visit a potential client's office, but the security guard won't let you in. You don't have a contact inside. What would you do? Feel free to share how you usually handle situations like this on the ground.",
            "evaluation_type": "Q1_AccessStrategy"
        },
        {
            "question_number": 2,
            "question": "You're pitching a company and they say, 'Your products are more expensive than our current supplier.' How would you respond? You can also tell us what's worked for you in the past when facing price objections.",
            "evaluation_type": "Q3_PriceObjections"
        },
        {
            "question_number": 3,
            "question": "It's been a tough week — lots of rejections, and you haven't closed any deals. It's Friday evening. What would you do to finish strong? You can also tell us how you usually stay motivated during slow weeks.",
            "evaluation_type": "Q4_MotivationAndResilience"
        },
        {
            "question_number": 4,
            "question": "A client says, 'We've been working with the same supplier for 10 years. Why should we switch to you?' How would you build trust and make them consider Sai Marketing?",
            "evaluation_type": "Q2_ClientTrustBuilding"
        },
        {
            "question_number": 5,
            "question": "You meet someone who doesn't know Sai Marketing. They ask, 'What do you do?' Explain our company and what we offer in under 30 seconds — just like you would in a real sales conversation.",
            "evaluation_type": "Q5_CompanyPitch"
        }
    ],
    "evaluation_rubric": {
        "Q1_AccessStrategy": {
            "prompt_instructions": "Evaluate based on candidate's proactive approach and resourcefulness in gaining access to a client's office without a contact.",
            "scoring_logic": {
                "base_scoring": [
                    {"threshold": 0, "score": 1, "criteria": "No clear strategy or gives up quickly."},
                    {"threshold": 1, "score": 2, "criteria": "Mentions one basic strategy (e.g., waiting)."},
                    {"threshold": 2, "score": 3, "criteria": "Suggests multiple standard strategies (e.g., leaving card, asking security for info)."},
                    {"threshold": 3, "score": 4, "criteria": "Demonstrates creative problem-solving or detailed plan (e.g., using LinkedIn on spot, finding nearby cafe)."},
                    {"threshold": 4, "score": 5, "criteria": "Highly resourceful, persistent, and demonstrates understanding of ground-level sales tactics (e.g., 'selling' to the guard, finding another entry point, leveraging public information)."}
                ],
                "keywords_proactive": ["find", "research", "ask", "observe", "creative", "alternative"],
                "notes": "Assesses initiative and practical field sales tactics."
            }
        },
        "Q2_ClientTrustBuilding": {
            "prompt_instructions": "Evaluate candidate's ability to build trust and persuade a client to switch from a long-term supplier to Sai Marketing.",
            "scoring_logic": {
                "base_scoring": [
                    {"threshold": 0, "score": 1, "criteria": "Focuses solely on features or negative comparisons."},
                    {"threshold": 1, "score": 2, "criteria": "Mentions 'value' but vaguely, or offers discounts immediately."},
                    {"threshold": 2, "score": 3, "criteria": "Attempts to understand current pain points, but lacks concrete strategy."},
                    {"threshold": 3, "score": 4, "criteria": "Focuses on understanding client's current situation, highlights differentiation, and offers a clear next step (e.g., pilot, case study)."},
                    {"threshold": 4, "score": 5, "criteria": "Emphasizes understanding client's specific needs, building rapport, demonstrating ROI/unique benefits, and creating a compelling reason to change through a structured approach (e.g., discovery call, competitive analysis, testimonials)."}
                ],
                "keywords_trust": ["understand", "listen", "rapport", "trust", "relationship", "pain points", "solution", "benefits", "ROI"],
                "notes": "Focuses on consultative selling and trust-building."
            }
        },
        "Q3_PriceObjections": {
            "prompt_instructions": "Evaluate based on the sequence of response to price objections, focusing on value communication for Sai Marketing's offerings.",
            "scoring_logic": {
                "response_sequences": [
                    {"sequence": ["discount", "price"], "score": 1, "criteria": "Immediately mentions discount/matching price"},
                    {"sequence": ["price", "value"], "score": 2, "criteria": "Mentions price then talks about value"},
                    {"sequence": ["value", "price"], "score": 3, "criteria": "Mentions value then addresses price"},
                    {"sequence": ["questions", "value"], "score": 4, "criteria": "Asks questions then discusses value"},
                    {"sequence": ["questions", "value", "alternatives"], "score": 5, "criteria": "Asks questions + discusses value + offers alternatives"}
                ],
                "value_keywords": ["ROI", "return", "benefit", "savings", "efficiency", "productivity", "outcome", "investment", "worth"],
                "question_indicators": ["let me understand", "can you tell me", "what would", "how much", "why is"],
                "notes": "Analyzes the flow of the answer in handling price objections."
            }
        },
        "Q4_MotivationAndResilience": {
            "prompt_instructions": "Evaluate candidate's approach to staying motivated and finishing strong during a tough sales week, demonstrating grit and drive.",
            "scoring_logic": {
                "base_scoring": [
                    {"threshold": 0, "score": 1, "criteria": "Suggests giving up or waiting for next week."},
                    {"threshold": 1, "score": 2, "criteria": "Mentions basic self-care (e.g., rest) without action."},
                    {"threshold": 2, "score": 3, "criteria": "Identifies some actionable steps but lacks specific detail or proactive mindset (e.g., 'try harder')."},
                    {"threshold": 3, "score": 4, "criteria": "Describes concrete actions to turn the week around (e.g., review pipeline, make more calls, connect with team for motivation)."},
                    {"threshold": 4, "score": 5, "criteria": "Demonstrates strong resilience, proactive measures, and a systematic approach to overcoming setbacks, including self-motivation strategies and learning from rejection."}
                ],
                "keywords_proactive": ["plan", "re-evaluate", "strategize", "learn", "analyse", "next steps"],
                "keywords_resilience": ["grit", "persistence", "bounce back", "motivation", "stay positive"],
                "notes": "Assesses resilience, drive, and proactive problem-solving."
            }
        },
        "Q5_CompanyPitch": {
            "prompt_instructions": "Evaluate candidate's ability to clearly and concisely explain Sai Marketing and its offerings under 30 seconds, demonstrating communication skills.",
            "scoring_logic": {
                "timing": [
                    {"range": [0, 14], "score": 1, "criteria": "Under 15 seconds"},
                    {"range": [15, 30], "score": 5, "criteria": "15-30 seconds (Optimal)"},
                    {"range": [31, 60], "score": 4, "criteria": "30-60 seconds"},
                    {"range": [61, 1000], "score": 2, "criteria": "Over 60 seconds"}
                ],
                "content_requirements": [
                    {"description": "Clear identification of what Sai Marketing *does* (e.g., 'digital marketing solutions')."},
                    {"description": "Identification of *who* they serve (e.g., 'businesses looking to grow online')."},
                    {"description": "Unique Selling Proposition/Benefit (e.g., 'drive measurable results')."}
                ],
                "min_content_elements": 2,
                "cap_if_missing_elements": 3,
                "notes": "Assesses conciseness, clarity, and ability to articulate value."
            }
        },
        "q_final":{
            "prompt_instructions": "Evaluate based on specificity in self-reflection and a growth mindset.",
            "scoring_logic": {
                "specificity_detection": [
                    {"type": "generic", "score": 1, "criteria": "All generic statements"},
                    {"type": "some_details", "score": 2, "criteria": "Some specific details"},
                    {"type": "1-2_examples", "score": 3, "criteria": "1-2 specific examples"},
                    {"type": "2-3_examples", "score": 4, "criteria": "2-3 specific examples"},
                    {"type": "multiple_detailed_examples", "score": 5, "criteria": "Multiple detailed examples"}
                ],
                "specificity_indicators": ["numbers", "company names", "specific situations", "metrics", "timeframes", "dollar amounts", "percentages"],
                "bonuses": [
                    {"type": "company_research", "score": 1, "keywords": ["Zenoti", "wellness industry", "competitors"]},
                    {"type": "growth_language", "score": 1, "keywords": ["learn", "improve", "develop", "coach", "feedback"]}
                ],
                "cap": 5,
                "notes": "Assesses the depth of self-reflection and openness to growth."
            }
        },
    },
    "trait_rubric": {
        "Resilience": {
            "prompt_instructions": "Evaluated primarily from Q3. Look for action-focused, systematic approach without blame.",
            "criteria": [
                {"score": 5, "description": "No blame language + action-focused + systematic approach"},
                {"score": 4, "description": "Mostly positive + some clear actions"},
                {"score": 3, "description": "Mixed tone but has plan"},
                {"score": 2, "description": "Some negativity but still tries"},
                {"score": 1, "description": "Blame language or gives up"}
            ],
            "source_question": 3,
            "keywords_positive": ["action-focused", "systematic approach"],
            "keywords_negative": ["blame", "fault", "unfair", "impossible", "won't work"]
        },
        "Drive": {
            "prompt_instructions": "Evaluated from Q2 and Q3. Look for fighting for value, urgency, and multiple tactics.",
            "criteria": [
                {"score": 5, "description": "Fights for value + shows urgency + multiple tactics"},
                {"score": 4, "description": "Defends value + some urgency"},
                {"score": 3, "description": "Basic value defense + standard effort"},
                {"score": 2, "description": "Weak value defense + low energy"},
                {"score": 1, "description": "Quick to discount + passive approach"}
            ],
            "source_question": [2, 3],
            "keywords_positive": ["fights for value", "urgency", "multiple tactics"],
            "keywords_negative": ["quick to discount", "passive approach"]
        },
        "Coachability": {
            "prompt_instructions": "Evaluated from Q6. Look for self-awareness, growth mindset, and admitting gaps.",
            "criteria": [
                {"score": 5, "description": "High self-awareness + growth mindset + admits gaps"},
                {"score": 4, "description": "Good self-awareness + learning orientation"},
                {"score": 3, "description": "Some self-awareness + open to feedback"},
                {"score": 2, "description": "Limited self-awareness + defensive"},
                {"score": 1, "description": "No self-awareness + know-it-all attitude"}
            ],
            "source_question": 6,
            "keywords_positive": ["self-awareness", "growth mindset", "admits gaps", "learn", "improve", "develop", "coach", "feedback"],
            "keywords_negative": ["defensive", "know-it-all attitude"]
        }
    },
    "overall_decision_thresholds": [
        {"score_range": [30, 40], "decision": "Strong Hire", "action": "Move to next round"},
        {"score_range": [22, 29], "decision": "Maybe", "action": "Human review recommended"},
        {"score_range": [15, 21], "decision": "Weak", "action": "Likely reject, but flag for review"},
        {"score_range": [0, 14], "decision": "Reject", "action": "Auto-reject"}
    ]
}

# Zenoti Sales Development Representative Configuration (NEW)
ZENOTI_SALES_DEVELOPMENT_REPRESENTATIVE_CONFIG = {
    "job_id": JOB_ID_ZENOTI_SALES_DEVELOPMENT_REPRESENTATIVE,
    "job_role": "Zenoti Sales Development Representative",
    "interview_questions": [
        {
            "question_number": 1,
            "question": "You need to reach the Operations Manager at a 15-location spa chain. You have no warm introduction. How would you get their attention?",
            "evaluation_type": "Q1_ProspectingApproach"
        },
        {
            "question_number": 2,
            "question": "You've called a prospect 3 times and emailed twice. They finally pick up and say 'I'm not interested, stop calling me.' How do you respond?",
            "evaluation_type": "Q2_HandlingRejection"
        },
        {
            "question_number": 3,
            "question": "A spa owner says 'Sure, I'll take a meeting to hear about your software.' What questions would you ask to qualify if this is worth your AE's time?",
            "evaluation_type": "Q3_QualifyingProspects"
        },
        {
            "question_number": 4,
            "question": "A prospect says 'This sounds interesting but we're not looking to change systems this year.' How do you create urgency without being pushy?",
            "evaluation_type": "Q4_CreatingUrgency"
        },
        {
            "question_number": 5,
            "question": "Tell me about a time you had to learn something completely new. How did you approach it and what kept you motivated?",
            "evaluation_type": "Q5_LearningAndMotivation"
        }
    ],
    "evaluation_rubric": {
        "Q1_ProspectingApproach": {
            "prompt_instructions": "Evaluate based on the number and creativity of outreach methods mentioned, and demonstration of personalization/value.",
            "scoring_logic": {
                "base_scoring": [
                    {"threshold": 1, "score": 1, "criteria": "1 method"},
                    {"threshold": 2, "score": 2, "criteria": "2 methods"},
                    {"threshold": 3, "score": 3, "criteria": "3 methods"},
                    {"threshold": 4, "score": 4, "criteria": "4 methods"},
                    {"threshold": 5, "score": 5, "criteria": "5+ methods"}
                ],
                "methods_to_detect": ["Email", "phone", "LinkedIn", "social media", "referrals", "events", "direct mail", "company research", "mutual connections"],
                "bonuses": [
                    {"type": "personalization_research", "score": 1, "keywords": ["personalization", "research", "tailor", "customize", "specific", "background"]},
                    {"type": "value_benefit", "score": 1, "keywords": ["value", "benefit", "solution", "ROI", "problem", "solve"]}
                ],
                "cap": 5,
                "penalties": [
                    {"keywords": ["showing up unannounced", "being pushy", "harass", "force"], "score": -1}
                ],
                "notes": "Assesses creativity and professionalism in prospecting."
            }
        },
        "Q2_HandlingRejection": {
            "prompt_instructions": "Evaluate how the candidate handles direct rejection, focusing on professionalism, empathy, and ability to pivot.",
            "scoring_logic": {
                "response_approach": [
                    {"score": 1, "criteria": "Gets defensive or argues"},
                    {"score": 2, "criteria": "Immediately gives up"},
                    {"score": 3, "criteria": "Acknowledges and offers alternative"},
                    {"score": 4, "criteria": "Acknowledges + asks permission + offers value"},
                    {"score": 5, "criteria": "Acknowledges + asks why + offers specific value + future follow-up"}
                ],
                "positive_phrases": ["understand", "respect", "appreciate", "sorry", "may I ask", "would it help if", "what if"],
                "negative_phrases": ["but", "however", "you're wrong", "you need", "let me tell you"],
                "bonuses": [
                    {"type": "ask_why", "score": 1, "keywords": ["why", "reason", "understand"]}
                ],
                "cap": 5,
                "notes": "Focuses on resilience and professional communication under pressure."
            }
        },
        "Q3_QualifyingProspects": {
            "prompt_instructions": "Evaluate the number and relevance of qualifying questions asked to determine if a prospect is a good fit.",
            "scoring_logic": {
                "base_scoring": [
                    {"threshold": 0, "score": 1, "criteria": "0-1 questions"},
                    {"threshold": 2, "score": 2, "criteria": "2 questions"},
                    {"threshold": 3, "score": 3, "criteria": "3 questions"},
                    {"threshold": 4, "score": 4, "criteria": "4 questions"},
                    {"threshold": 5, "score": 5, "criteria": "5+ questions"}
                ],
                "qualifying_areas_to_detect": ["Budget", "authority", "need", "timeline", "current solution", "decision process", "pain points", "company size", "locations"],
                "question_indicators": ["what", "how", "when", "who", "why", "tell me about", "walk me through"],
                "bonuses": [
                    {"type": "decision_making_process", "score": 1, "keywords": ["decision-making", "process", "approves", "stakeholders"]},
                    {"type": "budget_timeline", "score": 1, "keywords": ["budget", "timeline", "when", "investment", "cost"]}
                ],
                "cap": 5,
                "notes": "Assesses discovery and sales qualification skills."
            }
        },
        "Q4_CreatingUrgency": {
            "prompt_instructions": "Evaluate how the candidate creates urgency without being pushy, focusing on problem-solving and value.",
            "scoring_logic": {
                "approach_quality": [
                    {"score": 1, "criteria": "No clear strategy"},
                    {"score": 2, "criteria": "Basic attempt at urgency"},
                    {"score": 3, "criteria": "Asks discovery questions first"},
                    {"score": 4, "criteria": "Asks questions + creates business case"},
                    {"score": 5, "criteria": "Asks questions + business case + offers pilot/trial"}
                ],
                "discovery_words": ["what's driving", "what happens if", "what's the cost of", "how much time", "what would change"],
                "urgency_words": ["pilot", "trial", "small start", "quick win", "immediate impact", "competitive advantage", "missed opportunity"],
                "penalties": [
                    {"keywords": ["high-pressure tactics", "limited time offers", "artificial deadlines", "buy now", "expire"], "score": -2, "min_score": 1}
                ],
                "notes": "Assesses persuasive skills and ability to align solutions with business needs."
            }
        },
        "Q5_LearningAndMotivation": {
            "prompt_instructions": "Evaluate the specificity of the learning example and demonstration of motivation and self-directed learning.",
            "scoring_logic": {
                "specificity_of_example": [
                    {"score": 1, "criteria": "No specific example"},
                    {"score": 2, "criteria": "Vague example"},
                    {"score": 3, "criteria": "Some specific details"},
                    {"score": 4, "criteria": "Detailed example with context"},
                    {"score": 5, "criteria": "Detailed example with outcome and lessons"}
                ],
                "specificity_indicators": ["timeframes", "specific skills/topics", "measurable outcomes", "challenges faced", "methods used"],
                "bonuses": [
                    {"type": "persistence", "score": 1, "keywords": ["persistence", "difficulty", "struggle", "overcame", "kept going"]},
                    {"type": "self_directed_learning", "score": 1, "keywords": ["self-taught", "researched", "online course", "books", "mentors", "proactive"]}
                ],
                "cap": 5,
                "notes": "Assesses adaptability, initiative, and drive."
            }
        },
        "q_final":{
            "prompt_instructions": "Evaluate based on specificity in self-reflection and a growth mindset.",
            "scoring_logic": {
                "specificity_detection": [
                    {"type": "generic", "score": 1, "criteria": "All generic statements"},
                    {"type": "some_details", "score": 2, "criteria": "Some specific details"},
                    {"type": "1-2_examples", "score": 3, "criteria": "1-2 specific examples"},
                    {"type": "2-3_examples", "score": 4, "criteria": "2-3 specific examples"},
                    {"type": "multiple_detailed_examples", "score": 5, "criteria": "Multiple detailed examples"}
                ],
                "specificity_indicators": ["numbers", "company names", "specific situations", "metrics", "timeframes", "dollar amounts", "percentages"],
                "bonuses": [
                    {"type": "company_research", "score": 1, "keywords": ["Zenoti", "wellness industry", "competitors"]},
                    {"type": "growth_language", "score": 1, "keywords": ["learn", "improve", "develop", "coach", "feedback"]}
                ],
                "cap": 5,
                "notes": "Assesses the depth of self-reflection and openness to growth."
            }
        },
    },
    "trait_rubric": {
        "Resilience": {
            "prompt_instructions": "Evaluated from Q2 (Handling Rejection) + Q5 (Learning & Motivation).",
            "criteria": [
                {"score": 5, "description": "Handles rejection professionally + shows persistence in learning example"},
                {"score": 4, "description": "Good rejection handling + some persistence shown"},
                {"score": 3, "description": "Adequate rejection response + basic resilience"},
                {"score": 2, "description": "Weak rejection handling + limited resilience examples"},
                {"score": 1, "description": "Poor rejection response + gives up easily"}
            ],
            "source_question": [2, 5],
            "notes": "Reflects ability to bounce back from setbacks."
        },
        "Drive/Hunger": {
            "prompt_instructions": "Evaluated from Q1 (Prospecting Approach) + Q4 (Creating Urgency).",
            "criteria": [
                {"score": 5, "description": "Multiple creative prospecting methods + strong urgency creation"},
                {"score": 4, "description": "Good prospecting variety + decent urgency approach"},
                {"score": 3, "description": "Basic prospecting + standard urgency tactics"},
                {"score": 2, "description": "Limited prospecting creativity + weak urgency"},
                {"score": 1, "description": "Minimal effort + no real urgency creation"}
            ],
            "source_question": [1, 4],
            "notes": "Indicates proactive behavior and motivation to succeed."
        },
        "Coachability": {
            "prompt_instructions": "Evaluated from Q5 (Learning & Motivation) + Overall interview demeanor.",
            "criteria": [
                {"score": 5, "description": "Shows growth mindset + asks clarifying questions + admits learning needs"},
                {"score": 4, "description": "Good learning example + some growth orientation"},
                {"score": 3, "description": "Basic learning story + open to feedback"},
                {"score": 2, "description": "Limited learning example + somewhat defensive"},
                {"score": 1, "description": "No clear learning + defensive attitude"}
            ],
            "source_question": 5, # Primary source for this trait
            "notes": "Assesses openness to feedback and self-improvement."
        }
    },
    "overall_decision_thresholds": [
        {"score_range": [32, 40], "decision": "Strong Hire", "action": "Move to next round"},
        {"score_range": [25, 31], "decision": "Good", "action": "Likely hire with coaching"},
        {"score_range": [18, 24], "decision": "Maybe", "action": "Human review recommended"},
        {"score_range": [12, 17], "decision": "Weak", "action": "Likely reject"},
        {"score_range": [0, 11], "decision": "Reject", "action": "Auto-reject"}
    ]
}
CLICKPOST_CONFIG={
    "job_id": JOB_ID_CLICKPOST,
    "job_role": "Senior Account Executive, Clickpost",
    "interview_questions": [
        {
        "question_number": 1,
        "question": "You’re selling a product in a completely new industry. Walk me through how you’d get up to speed quickly — and how that helps you run high-value discovery calls.",
        "evaluation_type": "Q1_Learning_Agility"
    },
    {
        "question_number": 2,
        "question": "Could you tell me about a cold prospect you engaged successfully. What exactly did you say or do that got them to respond?",
        "evaluation_type": "Q2_Cold_Prospecting"
    },
    {
        "question_number": 3,
        "question": "Pick any B2B product you’ve sold. How did you move beyond surface-level problems to uncover the real drivers behind the client’s interest?",
        "evaluation_type": "Q3_Discovery_Depth"
    },
    {
        "question_number": 4,
        "question": "You’re pitching an AI logistics platform to a VP of Operations who’s short on time. In 2–3 sentences, how would you explain its value in a way that lands?",
        "evaluation_type": "Q4_Explaining_Complexity"
    },
    {
        "question_number": 5,
        "question": "Tell me about a deal or sales cycle that dragged on much longer than expected. What kept you going, and how did you stay on it?",
        "evaluation_type": "Q5_Objection_Handling"
    },
    {
        "question_number": 6,
        "question": "What’s prompting you to explore new opportunities right now? And in your next role, what are you hoping to learn, do, or achieve that you haven’t yet?",
        "evaluation_type": "Q6_Grit_Follow_Through"
    },

    
    ],
    "evaluation_rubric": {
    "Q1_Learning_Agility": {
        "prompt_instructions": "Evaluate based on structured learning, curiosity, and how the rep ties learning to discovery.",
        "scoring_logic": {
            "base_scoring": [
                {"threshold": 0, "score": 1, "criteria": "No real learning plan. Vague or generic response."},
                {"threshold": 1, "score": 2, "criteria": "Mentions documentation or passive learning. Some curiosity shown."},
                {"threshold": 2, "score": 3, "criteria": "Structured learning mentioned, basic discovery improvement intent."},
                {"threshold": 3, "score": 4, "criteria": "Shows proactive learning strategy with links to discovery."},
                {"threshold": 4, "score": 5, "criteria": "Clear, structured process tied to improving discovery with examples."}
            ],
            "keywords": ["learn", "training", "discovery", "documentation", "curious", "research", "plan", "improve"]
        }
    },
    "Q2_Cold_Prospecting": {
        "prompt_instructions": "Evaluate based on personalization, timing, and precision in outreach examples.",
        "scoring_logic": {
            "base_scoring": [
                {"threshold": 0, "score": 1, "criteria": "No clear example. Just says 'I followed up.'"},
                {"threshold": 1, "score": 2, "criteria": "Mentions basic personalization or timing. Example lacks depth."},
                {"threshold": 2, "score": 3, "criteria": "Specific outreach with 1-2 prospect-specific elements."},
                {"threshold": 3, "score": 4, "criteria": "Well-crafted outreach with timing or channel considerations."},
                {"threshold": 4, "score": 5, "criteria": "Detailed outreach strategy, tailored messaging, and outcome-focused."}
            ],
            "keywords": ["personalized", "outreach", "follow-up", "timing", "email", "LinkedIn", "message", "pain point"]
        }
    },
    "Q3_Discovery_Depth": {
        "prompt_instructions": "Evaluate based on empathy, probing questions, and ability to reframe client needs.",
        "scoring_logic": {
            "base_scoring": [
                {"threshold": 0, "score": 1, "criteria": "Only asks surface-level or scripted questions."},
                {"threshold": 1, "score": 2, "criteria": "Some probing beyond the surface, lacks structure."},
                {"threshold": 2, "score": 3, "criteria": "Probes with intent. Uncovers key needs but lacks insight."},
                {"threshold": 3, "score": 4, "criteria": "Identifies deeper needs and buyer motivations."},
                {"threshold": 4, "score": 5, "criteria": "Reframes client’s problem with insight. High empathy and depth."}
            ],
            "keywords": ["problem", "challenge", "why", "need", "goal", "impact", "business", "reframe", "dig deeper"]
        }
    },
    "Q4_Explaining_Complexity": {
        "prompt_instructions": "Evaluate based on ability to explain clearly, avoid jargon, and simplify for a business audience.",
        "scoring_logic": {
            "base_scoring": [
                {"threshold": 0, "score": 1, "criteria": "Rambling, jargon-heavy, or hard to follow."},
                {"threshold": 1, "score": 2, "criteria": "Clear articulation of product but technical or vague."},
                {"threshold": 2, "score": 3, "criteria": "Explains clearly and simplifies some features."},
                {"threshold": 3, "score": 4, "criteria": "Clear explanation focused on business value, not features."},
                {"threshold": 4, "score": 5, "criteria": "Crisp, outcome-driven pitch that resonates with VP-level audience."}
            ],
            "keywords": ["value", "outcome", "simplify", "explain", "business impact", "clarity", "jargon", "pitch"]
        }
    },
    "Q5_Objection_Handling": {
        "prompt_instructions": "Evaluate based on composure, logic, and ability to reframe tied to business outcomes.",
        "scoring_logic": {
            "base_scoring": [
                {"threshold": 0, "score": 1, "criteria": "Defensive or reactive. Offers a discount immediately."},
                {"threshold": 1, "score": 2, "criteria": "Attempts logical response, misses emotional tone or goal."},
                {"threshold": 2, "score": 3, "criteria": "Addresses concern logically, partial link to business goals."},
                {"threshold": 3, "score": 4, "criteria": "Calm, structured response with outcome framing."},
                {"threshold": 4, "score": 5, "criteria": "Handles pushback with composure, reframes using client goals."}
            ],
            "keywords": ["objection", "concern", "pushback", "handle", "understand", "business", "discount", "reframe"]
        }
    },
    "Q6_Grit_Follow_Through": {
        "prompt_instructions": "Evaluate based on persistence, personal ownership, and use of multiple strategies.",
        "scoring_logic": {
            "base_scoring": [
                {"threshold": 0, "score": 1, "criteria": "Gives up or blames external blockers."},
                {"threshold": 1, "score": 2, "criteria": "Follows up once or twice, limited strategy."},
                {"threshold": 2, "score": 3, "criteria": "Shows persistence with a couple of approaches."},
                {"threshold": 3, "score": 4, "criteria": "Tries multiple strategies and follows through consistently."},
                {"threshold": 4, "score": 5, "criteria": "Highly persistent. Shows ownership and creative strategies over time."}
            ],
            "keywords": ["follow up", "try again", "persistent", "strategy", "ownership", "blocker", "commitment", "email", "call"]
        }
    }
},
        "q_final":{
            "prompt_instructions": "Evaluate based on specificity in self-reflection and a growth mindset.",
            "scoring_logic": {
                "specificity_detection": [
                    {"type": "generic", "score": 1, "criteria": "All generic statements"},
                    {"type": "some_details", "score": 2, "criteria": "Some specific details"},
                    {"type": "1-2_examples", "score": 3, "criteria": "1-2 specific examples"},
                    {"type": "2-3_examples", "score": 4, "criteria": "2-3 specific examples"},
                    {"type": "multiple_detailed_examples", "score": 5, "criteria": "Multiple detailed examples"}
                ],
                "specificity_indicators": ["numbers", "company names", "specific situations", "metrics", "timeframes", "dollar amounts", "percentages"],
                "bonuses": [
                    {"type": "company_research", "score": 1, "keywords": ["Zenoti", "wellness industry", "competitors"]},
                    {"type": "growth_language", "score": 1, "keywords": ["learn", "improve", "develop", "coach", "feedback"]}
                ],
                "cap": 5,
                "notes": "Assesses the depth of self-reflection and openness to growth."
            }
        },
    "trait_rubric": {
        "Grit": {
            "prompt_instructions": "Evaluated from Q1 + Q4. Look for persistence and action under pressure.",
            "criteria": [
                {"score": 5, "description": "Creative tactics + urgency + finishes strong"},
                {"score": 4, "description": "Multiple tactics + solid follow-through"},
                {"score": 3, "description": "Basic persistence and intent"},
                {"score": 2, "description": "Mild effort, limited urgency"},
                {"score": 1, "description": "Gives up or disengaged"}
            ],
            "source_question": [1, 4]
        },
        "Adaptability": {
            "prompt_instructions": "Evaluated from Q2. Look for reflection, signal interpretation, and fast adjustment.",
            "criteria": [
                {"score": 5, "description": "Rapid learning + clear buyer signal plan"},
                {"score": 4, "description": "Learns and adapts clearly"},
                {"score": 3, "description": "Some self-awareness"},
                {"score": 2, "description": "Struggles with adjustment"},
                {"score": 1, "description": "Defensive or unaware"}
            ],
            "source_question": 2
        },
        "Coachability": {
            "prompt_instructions": "Evaluated from Q5. Look for humility, learning mindset, and openness.",
            "criteria": [
                {"score": 5, "description": "Mentions learning, feedback, and growth directly"},
                {"score": 4, "description": "Shows willingness to adapt"},
                {"score": 3, "description": "Neutral or general positivity"},
                {"score": 2, "description": "Mild resistance or vague"},
                {"score": 1, "description": "Closed off or arrogant"}
            ],
            "source_question": 5
        }
    },
    "overall_decision_thresholds": [
        {"score_range": [40, 50], "decision": "Proceed", "action": "Move to next round"},
        {"score_range": [30, 39], "decision": "Maybe", "action": "Human review recommended"},
        {"score_range": [20, 29], "decision": "Weak", "action": "Likely reject, but flag for review"},
        {"score_range": [0, 19], "decision": "Do not proceed", "action": "Auto-reject"}
    ]
}
DEMO_CONFIG = {
    "job_id": JOB_ID_DEMO,
    "job_role": "Sales, Demo",
    "interview_questions": [
        {
            "question_number": 1,
            "question": "Can you tell me about a time when you had to learn something completely new? How did you approach it?",
            "evaluation_type": "Q1_Learning_Agility"
        },
        {
            "question_number": 2,
            "question": "Describe a time when you initiated contact with someone you didn’t know. How did you start the conversation?",
            "evaluation_type": "Q2_Cold_Prospecting"
        },
        {
            "question_number": 3,
            "question": "When speaking with someone about a need or problem, how do you make sure you understand the full picture?",
            "evaluation_type": "Q3_Discovery_Depth"
        },
        {
            "question_number": 4,
            "question": "Imagine you need to explain a complicated idea to someone unfamiliar with the topic. How would you do it?",
            "evaluation_type": "Q4_Explaining_Complexity"
        },
        {
            "question_number": 5,
            "question": "Tell me about a situation where progress was slow or challenging. How did you stay committed?",
            "evaluation_type": "Q5_Objection_Handling"
        },
        {
            "question_number": 6,
            "question": "Why are you looking for new opportunities, and what do you hope to achieve in your next role?",
            "evaluation_type": "Q6_Grit_Follow_Through"
        },

    ],
    "evaluation_rubric": {
    "Q1_Learning_Agility": {
        "prompt_instructions": "Evaluate based on structured learning, curiosity, and how the rep ties learning to discovery.",
        "scoring_logic": {
            "base_scoring": [
                {"threshold": 0, "score": 1, "criteria": "No real learning plan. Vague or generic response."},
                {"threshold": 1, "score": 2, "criteria": "Mentions documentation or passive learning. Some curiosity shown."},
                {"threshold": 2, "score": 3, "criteria": "Structured learning mentioned, basic discovery improvement intent."},
                {"threshold": 3, "score": 4, "criteria": "Shows proactive learning strategy with links to discovery."},
                {"threshold": 4, "score": 5, "criteria": "Clear, structured process tied to improving discovery with examples."}
            ],
            "keywords": ["learn", "training", "discovery", "documentation", "curious", "research", "plan", "improve"]
        }
    },
    "Q2_Cold_Prospecting": {
        "prompt_instructions": "Evaluate based on personalization, timing, and precision in outreach examples.",
        "scoring_logic": {
            "base_scoring": [
                {"threshold": 0, "score": 1, "criteria": "No clear example. Just says 'I followed up.'"},
                {"threshold": 1, "score": 2, "criteria": "Mentions basic personalization or timing. Example lacks depth."},
                {"threshold": 2, "score": 3, "criteria": "Specific outreach with 1-2 prospect-specific elements."},
                {"threshold": 3, "score": 4, "criteria": "Well-crafted outreach with timing or channel considerations."},
                {"threshold": 4, "score": 5, "criteria": "Detailed outreach strategy, tailored messaging, and outcome-focused."}
            ],
            "keywords": ["personalized", "outreach", "follow-up", "timing", "email", "LinkedIn", "message", "pain point"]
        }
    },
    "Q3_Discovery_Depth": {
        "prompt_instructions": "Evaluate based on empathy, probing questions, and ability to reframe client needs.",
        "scoring_logic": {
            "base_scoring": [
                {"threshold": 0, "score": 1, "criteria": "Only asks surface-level or scripted questions."},
                {"threshold": 1, "score": 2, "criteria": "Some probing beyond the surface, lacks structure."},
                {"threshold": 2, "score": 3, "criteria": "Probes with intent. Uncovers key needs but lacks insight."},
                {"threshold": 3, "score": 4, "criteria": "Identifies deeper needs and buyer motivations."},
                {"threshold": 4, "score": 5, "criteria": "Reframes client’s problem with insight. High empathy and depth."}
            ],
            "keywords": ["problem", "challenge", "why", "need", "goal", "impact", "business", "reframe", "dig deeper"]
        }
    },
    "Q4_Explaining_Complexity": {
        "prompt_instructions": "Evaluate based on ability to explain clearly, avoid jargon, and simplify for a business audience.",
        "scoring_logic": {
            "base_scoring": [
                {"threshold": 0, "score": 1, "criteria": "Rambling, jargon-heavy, or hard to follow."},
                {"threshold": 1, "score": 2, "criteria": "Clear articulation of product but technical or vague."},
                {"threshold": 2, "score": 3, "criteria": "Explains clearly and simplifies some features."},
                {"threshold": 3, "score": 4, "criteria": "Clear explanation focused on business value, not features."},
                {"threshold": 4, "score": 5, "criteria": "Crisp, outcome-driven pitch that resonates with VP-level audience."}
            ],
            "keywords": ["value", "outcome", "simplify", "explain", "business impact", "clarity", "jargon", "pitch"]
        }
    },
    "Q5_Objection_Handling": {
        "prompt_instructions": "Evaluate based on composure, logic, and ability to reframe tied to business outcomes.",
        "scoring_logic": {
            "base_scoring": [
                {"threshold": 0, "score": 1, "criteria": "Defensive or reactive. Offers a discount immediately."},
                {"threshold": 1, "score": 2, "criteria": "Attempts logical response, misses emotional tone or goal."},
                {"threshold": 2, "score": 3, "criteria": "Addresses concern logically, partial link to business goals."},
                {"threshold": 3, "score": 4, "criteria": "Calm, structured response with outcome framing."},
                {"threshold": 4, "score": 5, "criteria": "Handles pushback with composure, reframes using client goals."}
            ],
            "keywords": ["objection", "concern", "pushback", "handle", "understand", "business", "discount", "reframe"]
        }
    },
    "Q6_Grit_Follow_Through": {
        "prompt_instructions": "Evaluate based on persistence, personal ownership, and use of multiple strategies.",
        "scoring_logic": {
            "base_scoring": [
                {"threshold": 0, "score": 1, "criteria": "Gives up or blames external blockers."},
                {"threshold": 1, "score": 2, "criteria": "Follows up once or twice, limited strategy."},
                {"threshold": 2, "score": 3, "criteria": "Shows persistence with a couple of approaches."},
                {"threshold": 3, "score": 4, "criteria": "Tries multiple strategies and follows through consistently."},
                {"threshold": 4, "score": 5, "criteria": "Highly persistent. Shows ownership and creative strategies over time."}
            ],
            "keywords": ["follow up", "try again", "persistent", "strategy", "ownership", "blocker", "commitment", "email", "call"]
        }
    }
},
        "q_final":{
            "prompt_instructions": "Evaluate based on specificity in self-reflection and a growth mindset.",
            "scoring_logic": {
                "specificity_detection": [
                    {"type": "generic", "score": 1, "criteria": "All generic statements"},
                    {"type": "some_details", "score": 2, "criteria": "Some specific details"},
                    {"type": "1-2_examples", "score": 3, "criteria": "1-2 specific examples"},
                    {"type": "2-3_examples", "score": 4, "criteria": "2-3 specific examples"},
                    {"type": "multiple_detailed_examples", "score": 5, "criteria": "Multiple detailed examples"}
                ],
                "specificity_indicators": ["numbers", "company names", "specific situations", "metrics", "timeframes", "dollar amounts", "percentages"],
                "bonuses": [
                    {"type": "company_research", "score": 1, "keywords": ["Zenoti", "wellness industry", "competitors"]},
                    {"type": "growth_language", "score": 1, "keywords": ["learn", "improve", "develop", "coach", "feedback"]}
                ],
                "cap": 5,
                "notes": "Assesses the depth of self-reflection and openness to growth."
            }
        },
    "trait_rubric": {
        "Grit": {
            "prompt_instructions": "Evaluated from Q1 + Q4. Look for persistence and action under pressure.",
            "criteria": [
                {"score": 5, "description": "Creative tactics + urgency + finishes strong"},
                {"score": 4, "description": "Multiple tactics + solid follow-through"},
                {"score": 3, "description": "Basic persistence and intent"},
                {"score": 2, "description": "Mild effort, limited urgency"},
                {"score": 1, "description": "Gives up or disengaged"}
            ],
            "source_question": [1, 4]
        },
        "Adaptability": {
            "prompt_instructions": "Evaluated from Q2. Look for reflection, signal interpretation, and fast adjustment.",
            "criteria": [
                {"score": 5, "description": "Rapid learning + clear buyer signal plan"},
                {"score": 4, "description": "Learns and adapts clearly"},
                {"score": 3, "description": "Some self-awareness"},
                {"score": 2, "description": "Struggles with adjustment"},
                {"score": 1, "description": "Defensive or unaware"}
            ],
            "source_question": 2
        },
        "Coachability": {
            "prompt_instructions": "Evaluated from Q5. Look for humility, learning mindset, and openness.",
            "criteria": [
                {"score": 5, "description": "Mentions learning, feedback, and growth directly"},
                {"score": 4, "description": "Shows willingness to adapt"},
                {"score": 3, "description": "Neutral or general positivity"},
                {"score": 2, "description": "Mild resistance or vague"},
                {"score": 1, "description": "Closed off or arrogant"}
            ],
            "source_question": 5
        }
    },
    "overall_decision_thresholds": [
        {"score_range": [40, 50], "decision": "Proceed", "action": "Move to next round"},
        {"score_range": [30, 39], "decision": "Maybe", "action": "Human review recommended"},
        {"score_range": [20, 29], "decision": "Weak", "action": "Likely reject, but flag for review"},
        {"score_range": [0, 19], "decision": "Do not proceed", "action": "Auto-reject"}
    ]
}

ASTER_HEALTHCARE_CONFIG = {
    "job_id": JOB_ID_ASTER_HEALTHCARE,
    "job_role": "Healthcare Education Advisor",
    "interview_questions": [
        {
            "question_number": 1,
            "question": "Tell me about your approach to cold calling professionals who are typically very busy. How do you start, build and maintain rapport, what questions do you ask?",
            "evaluation_type": "Q1_ColdCallingRapport"
        },
        {
            "question_number": 2,
            "question": "Think back to a time you've helped someone make an important decision when they were unsure. How did you guide them?",
            "evaluation_type": "Q2_ConsultativeGuidance"
        },
        {
            "question_number": 3,
            "question": "Imagine a healthcare professional says 'I want to advance in my career but don't know which direction.' How would you counsel them about prospective courses?",
            "evaluation_type": "Q3_AdvisoryCounseling"
        },
        {
            "question_number": 4,
            "question": "Tell me about a time when a customer had a price or cost related objection, how did you handle it? Specify what was their objection and how did you address the objection?",
            "evaluation_type": "Q4_ObjectionHandling"
        },
        {
            "question_number": 5,
            "question": "This is your 6th call to a dental student you are trying to enroll in a course. In every call they say 'let me think about it.' What do you do?",
            "evaluation_type": "Q5_FollowUpStrategy"
        },
        {
            "question_number": 6,
            "question": "Tell me about a time when you faced multiple rejections in a row. How did you handle it?",
            "evaluation_type": "Q6_ResilienceExperience"
        },
        {
            "question_number": 7,
            "question": "Let’s say you have been making calls all morning and heard 15 straight 'no's today. You still have 50 more calls to make. How do you handle this?",
            "evaluation_type": "Q7_EnergyManagement"
        }
    ],
    "evaluation_rubric": {
        "Q1_ColdCallingRapport": {
            "prompt_instructions": "Evaluate based on ability to start a cold call, establish rapport, and ask discovery questions.",
            "scoring_logic": {
                "base_scoring": [
                    {"threshold": 0, "score": 1, "criteria": "No structure, generic opener"},
                    {"threshold": 1, "score": 2, "criteria": "Mentions rapport OR opener but little depth"},
                    {"threshold": 2, "score": 3, "criteria": "Clear opener + 1-2 rapport techniques"},
                    {"threshold": 3, "score": 4, "criteria": "Opener + rapport + at least 2 discovery questions"},
                    {"threshold": 4, "score": 5, "criteria": "Strong opener, rapport, discovery questions, empathy in tone"}
                ],
                "keywords_rapport": ["understand", "empathize", "respect your time", "curious", "value"],
                "question_indicators": ["can you tell me", "what’s your process", "how do you currently", "why is"],
                "notes": "Looks for confidence, empathy, and structured cold call flow."
            }
        },
        "Q2_ConsultativeGuidance": {
            "prompt_instructions": "Evaluate based on active listening and guiding process for decision-making.",
            "scoring_logic": {
                "specificity_detection": [
                    {"type": "generic", "score": 1, "criteria": "Very vague guidance"},
                    {"type": "basic_support", "score": 2, "criteria": "Offers encouragement but not structured"},
                    {"type": "some_structure", "score": 3, "criteria": "Provides some steps or considerations"},
                    {"type": "guided_process", "score": 4, "criteria": "Asks questions + provides structured framework"},
                    {"type": "consultative", "score": 5, "criteria": "Uses questions, frameworks, empathy, and guides to decision"}
                ],
                "bonuses": [
                    {"keywords": ["options", "pros and cons", "future goals", "support"], "score": 1}
                ],
                "notes": "Assesses patience, trustworthiness, and ability to guide uncertain individuals."
            }
        },
        "Q3_AdvisoryCounseling": {
            "prompt_instructions": "Evaluate based on needs analysis and advisory recommendations.",
            "scoring_logic": {
                "base_scoring": [
                    {"threshold": 0, "score": 1, "criteria": "No real guidance"},
                    {"threshold": 1, "score": 2, "criteria": "Generic advice without probing"},
                    {"threshold": 2, "score": 3, "criteria": "Asks basic questions + suggests broad options"},
                    {"threshold": 3, "score": 4, "criteria": "Conducts needs analysis + aligns with 1-2 course paths"},
                    {"threshold": 4, "score": 5, "criteria": "Thorough needs analysis + tailored, practical course suggestions"}
                ],
                "needs_keywords": ["goals", "skills", "interest", "career path", "specialization"],
                "advisory_keywords": ["recommend", "suggest", "best fit", "consider"],
                "notes": "Strong advisory answers should show helpfulness and analytical thinking."
            }
        },
        "Q4_ObjectionHandling": {
            "prompt_instructions": "Evaluate based on objection handling flow: listen, clarify, communicate value, resolve.",
            "scoring_logic": {
                "response_sequences": [
                    {"sequence": ["discount", "price"], "score": 1, "criteria": "Jumps to discount"},
                    {"sequence": ["price", "value"], "score": 2, "criteria": "Talks price then value"},
                    {"sequence": ["value", "price"], "score": 3, "criteria": "Leads with value then discusses price"},
                    {"sequence": ["questions", "value"], "score": 4, "criteria": "Asks clarifying questions + value discussion"},
                    {"sequence": ["questions", "value", "alternatives"], "score": 5, "criteria": "Clarifies, emphasizes value, and offers alternatives"}
                ],
                "value_keywords": ["ROI", "investment", "benefit", "long-term", "outcomes"],
                "question_indicators": ["can you share", "what makes", "help me understand"],
                "notes": "Assesses persistence and problem-solving approach."
            }
        },
        "Q5_FollowUpStrategy": {
            "prompt_instructions": "Evaluate based on creativity and persistence in handling repeated 'let me think about it.'",
            "scoring_logic": {
                "base_scoring": [
                    {"threshold": 0, "score": 1, "criteria": "Keeps repeating ask with no change"},
                    {"threshold": 1, "score": 2, "criteria": "Acknowledges delay but no strategy"},
                    {"threshold": 2, "score": 3, "criteria": "Suggests follow-up at specific time or shares info"},
                    {"threshold": 3, "score": 4, "criteria": "Offers new angle/content + specific timeline"},
                    {"threshold": 4, "score": 5, "criteria": "Uses multiple tactics (urgency, content, deadlines, alternative approaches)"}
                ],
                "bonuses": [
                    {"type": "strategic_content", "score": 1, "keywords": ["case study", "testimonial", "results"]}
                ],
                "notes": "Focuses on adaptability and strategic follow-up."
            }
        },
        "Q6_ResilienceExperience": {
            "prompt_instructions": "Evaluate based on reflection on rejection handling.",
            "scoring_logic": {
                "specificity_detection": [
                    {"type": "generic", "score": 1, "criteria": "Vague or avoidance"},
                    {"type": "some_details", "score": 2, "criteria": "Mentions emotions but no clear coping strategy"},
                    {"type": "basic_strategy", "score": 3, "criteria": "Shares simple resilience tactic"},
                    {"type": "structured_strategy", "score": 4, "criteria": "Mentions reflection + specific resilience methods"},
                    {"type": "growth_strategy", "score": 5, "criteria": "Demonstrates growth mindset + systematic resilience techniques"}
                ],
                "keywords_positive": ["learn", "bounce back", "improve", "adapt"],
                "notes": "Assesses mental toughness and growth mindset."
            }
        },
        "Q7_EnergyManagement": {
            "prompt_instructions": "Evaluate based on maintaining motivation during long stretches of rejection.",
            "scoring_logic": {
                "base_scoring": [
                    {"threshold": 0, "score": 1, "criteria": "Gives up / negative response"},
                    {"threshold": 1, "score": 2, "criteria": "Pushes through but no strategy"},
                    {"threshold": 2, "score": 3, "criteria": "Mentions break or mindset shift"},
                    {"threshold": 3, "score": 4, "criteria": "Uses specific techniques (music, affirmations, reframing)"},
                    {"threshold": 4, "score": 5, "criteria": "Has systematic methods for energy + motivation reset"}
                ],
                "keywords_positive": ["reset", "reframe", "stay motivated", "focus", "mindset"],
                "notes": "Assesses resilience and self-motivation."
            }
        },
        "q_final":{
            "prompt_instructions": "Evaluate based on specificity in self-reflection and a growth mindset.",
            "scoring_logic": {
                "specificity_detection": [
                    {"type": "generic", "score": 1, "criteria": "All generic statements"},
                    {"type": "some_details", "score": 2, "criteria": "Some specific details"},
                    {"type": "1-2_examples", "score": 3, "criteria": "1-2 specific examples"},
                    {"type": "2-3_examples", "score": 4, "criteria": "2-3 specific examples"},
                    {"type": "multiple_detailed_examples", "score": 5, "criteria": "Multiple detailed examples"}
                ],
                "specificity_indicators": ["numbers", "company names", "specific situations", "metrics", "timeframes", "dollar amounts", "percentages"],
                "bonuses": [
                    {"type": "company_research", "score": 1, "keywords": ["Zenoti", "wellness industry", "competitors"]},
                    {"type": "growth_language", "score": 1, "keywords": ["learn", "improve", "develop", "coach", "feedback"]}
                ],
                "cap": 5,
                "notes": "Assesses the depth of self-reflection and openness to growth."
            }
        },
    },
    "trait_rubric": {
        "Confidence": {
            "prompt_instructions": "Evaluated from Q1. Look for confident openers and proactive questions.",
            "criteria": [
                {"score": 5, "description": "Clear opener + confident, empathetic questions"},
                {"score": 3, "description": "Some confidence but hesitant"},
                {"score": 1, "description": "No confidence, generic approach"}
            ],
            "source_question": 1
        },
        "Empathy": {
            "prompt_instructions": "Evaluated from Q1 + Q2. Look for language showing understanding and care.",
            "criteria": [
                {"score": 5, "description": "Consistent empathetic language + adjusts to prospect"},
                {"score": 3, "description": "Some empathy but not consistent"},
                {"score": 1, "description": "No empathy or dismissive"}
            ],
            "source_question": [1, 2]
        },
        "Helpfulness": {
            "prompt_instructions": "Evaluated from Q2 + Q3. Look for consultative, tailored guidance.",
            "criteria": [
                {"score": 5, "description": "Deep consultative guidance with options"},
                {"score": 3, "description": "Some useful suggestions but generic"},
                {"score": 1, "description": "Not helpful or dismissive"}
            ],
            "source_question": [2, 3]
        },
        "Persistence": {
            "prompt_instructions": "Evaluated from Q4 + Q5. Look for proactive objection handling and creative follow-ups.",
            "criteria": [
                {"score": 5, "description": "Strong value defense + creative follow-up"},
                {"score": 3, "description": "Basic objection handling and persistence"},
                {"score": 1, "description": "Quick to give up or discount"}
            ],
            "source_question": [4, 5]
        },
        "Resilience": {
            "prompt_instructions": "Evaluated from Q6 + Q7. Look for systematic approach to rejection and motivation.",
            "criteria": [
                {"score": 5, "description": "Strong coping + energy reset strategies"},
                {"score": 3, "description": "Some strategies but not consistent"},
                {"score": 1, "description": "Negative or defeatist"}
            ],
            "source_question": [6, 7]
        }
    },
    "overall_decision_thresholds": [
        {"score_range": [31, 35], "decision": "Strong Hire", "action": "Move to next round"},
        {"score_range": [24, 30], "decision": "Maybe", "action": "Human review recommended"},
        {"score_range": [18, 23], "decision": "Weak", "action": "Likely reject, but flag for review"},
        {"score_range": [0, 17], "decision": "Reject", "action": "Auto-reject"}
    ]
}
GERMIN8_CONFIG = {
    "job_id": JOB_ID_GERMIN8,
    "job_role": "Enterprise Sales Representative",
    "interview_questions": [
        {
            "question_number": 1,
            "question": "You have 3 enterprise prospects: Company A: Big company, no software yet, just had issues with their current way of doing things. Company B: Medium-sized company, has basic tools, limited budget. Company C: Big company, their current contract is ending soon, looking at options. Who would you go after first and what would your outreach strategy be?",
            "evaluation_type": "Q1_ProspectPrioritization"
        },
        {
            "question_number": 2,
            "question": "Can you share an example of when you successfully prioritized prospects. What was your process and outcome?",
            "evaluation_type": "Q2_RealPrioritizationExample"
        },
        {
            "question_number": 3,
            "question": "A prospect tells you: 'We're doing everything manually right now. It takes forever and leadership wants better visibility, but we don't know what options are out there.' What questions would you ask them? How would you find out their budget, who makes the decisions, and when they want to make a change?",
            "evaluation_type": "Q3_DiscoveryQualification"
        },
        {
            "question_number": 4,
            "question": "Tell us about a time you were talking with a prospect and discovered something important that other salespeople probably missed. What did you find out and how did it help you with the sale?",
            "evaluation_type": "Q4_DiscoverySuccess"
        },
        {
            "question_number": 5,
            "question": "You are on a prospect call where they bring up two objections. 'How do I know this won't just sit there unused like the last software we bought?' and  'Budget is tight. Can we start smaller?' How would you address these two concerns while keeping the deal moving forward?",
            "evaluation_type": "Q5_TwoObjections"
        },
        {
            "question_number": 6,
            "question": " Explain what Germin8 does in 2-3 simple sentences, like you're telling a friend who has never heard of it. If you don't know Germin8 well, use your current company's product instead.",
            "evaluation_type": "Q6_ValueProposition"
        },
        {
            "question_number": 7,
            "question": "What's the toughest pushback or objection you've gotten from a customer in your sales career? How did you handle it?",
            "evaluation_type": "Q7_ChallengingObjection"
        },
        {
            "question_number": 8,
            "question": " Everyone loses deals sometimes. Tell us about one you lost that taught you something important. What would you do differently if you had that situation again?",
            "evaluation_type": "Q8_LearningFromLoss"
        }
    ],
    "evaluation_rubric": {
        "Q1_ProspectPrioritization": {
            "prompt_instructions": "Evaluate based on prioritization logic and outreach strategy.",
            "scoring_logic": {
                "prioritization": [
                    {"score": 1, "criteria": "States clear choice with basic reasoning"},
                    {"score": 2, "criteria": "Uses 2+ strategic factors (timing, budget readiness, competition, urgency)"}
                ],
                "outreach": [
                    {"score": 1, "criteria": "Mentions specific outreach method"},
                    {"score": 2, "criteria": "Personalizes outreach to chosen company's situation"}
                ],
                "notes": "Total possible: 4 points"
            }
        },
        "Q2_RealPrioritizationExample": {
            "prompt_instructions": "Evaluate based on example completeness and prioritization process.",
            "scoring_logic": {
                "example": [
                    {"score": 1, "criteria": "Specific situation with context"},
                    {"score": 2, "criteria": "Includes measurable outcome"}
                ],
                "process": [
                    {"score": 1, "criteria": "Mentions clear process/methodology"},
                    {"score": 2, "criteria": "Systematic approach with criteria/tools"}
                ],
                "red_flags": ["Vague example", "No measurable results", "No real example"],
                "notes": "Total possible: 4 points"
            }
        },
        "Q3_DiscoveryQualification": {
            "prompt_instructions": "Evaluate based on discovery questions and qualification coverage.",
            "scoring_logic": {
                "question_quality": [
                    {"score": 1, "criteria": "Asks about current process"},
                    {"score": 2, "criteria": "Asks about business impact/pain"},
                    {"score": 3, "criteria": "Asks about stakeholders/decision-makers"}
                ],
                "qualification": [
                    {"score": 1, "criteria": "Mentions budget AND timeline"}
                ],
                "bonuses": [
                    {"score": 0.5, "criteria": "Open-ended format"},
                    {"score": 0.5, "criteria": "Logical flow or follow-up strategy"}
                ],
                "red_flags": ["Closed-ended only", "Feature-focused questions", "No budget/timeline"],
                "notes": "Total possible: 4 points +1 bonus"
            }
        },
        "Q4_DiscoverySuccess": {
            "prompt_instructions": "Evaluate based on specific discovery example and impact.",
            "scoring_logic": {
                "example": [
                    {"score": 1, "criteria": "Specific situation with context"},
                    {"score": 2, "criteria": "Clear outcome/impact"}
                ],
                "discovery_quality": [
                    {"score": 1, "criteria": "Uncovered something others missed"},
                    {"score": 2, "criteria": "Demonstrates strategic thinking or deep listening"}
                ],
                "red_flags": ["Vague example", "No clear impact", "No specific discovery"],
                "notes": "Total possible: 4 points"
            }
        },
        "Q5_TwoObjections": {
            "prompt_instructions": "Evaluate based on handling adoption and budget objections.",
            "scoring_logic": {
                "adoption": [
                    {"score": 1, "criteria": "Acknowledges concern with empathy"},
                    {"score": 2, "criteria": "Provides solution (references, pilot, ROI, training)"}
                ],
                "budget": [
                    {"score": 1, "criteria": "Shows understanding of budget constraints"},
                    {"score": 2, "criteria": "Provides alternatives (phased, smaller scope, ROI justification)"}
                ],
                "red_flags": ["Dismissive tone", "Ignores one objection", "No solutions"],
                "notes": "Total possible: 4 points"
            }
        },
        "Q6_ValueProposition": {
            "prompt_instructions": "Evaluate based on clarity and business focus.",
            "scoring_logic": {
                "clarity": [
                    {"score": 1, "criteria": "Simple language"},
                    {"score": 2, "criteria": "Within 2-3 sentences"}
                ],
                "business_focus": [
                    {"score": 1, "criteria": "Mentions business benefit"},
                    {"score": 2, "criteria": "Outcome-focused language"}
                ],
                "red_flags": ["Too technical", "Feature-only", "Too long"],
                "notes": "Total possible: 4 points"
            }
        },
        "Q7_ChallengingObjection": {
            "prompt_instructions": "Evaluate based on objection example and problem-solving quality.",
            "scoring_logic": {
                "example": [
                    {"score": 1, "criteria": "Specific objection with context"},
                    {"score": 2, "criteria": "Clear resolution and outcome"}
                ],
                "problem_solving": [
                    {"score": 1, "criteria": "Shows strategy used"},
                    {"score": 2, "criteria": "Demonstrates persistence or creative thinking"}
                ],
                "red_flags": ["Blames customer", "No resolution", "No specific example"],
                "notes": "Total possible: 4 points"
            }
        },
        "Q8_LearningFromLoss": {
            "prompt_instructions": "Evaluate based on accountability and growth.",
            "scoring_logic": {
                "accountability": [
                    {"score": 1, "criteria": "Takes responsibility"},
                    {"score": 2, "criteria": "Identifies specific mistakes"}
                ],
                "growth": [
                    {"score": 1, "criteria": "Clear lesson learned"},
                    {"score": 2, "criteria": "Changed approach going forward"}
                ],
                "red_flags": ["Blames others", "No learning", "Defensive"],
                "notes": "Total possible: 4 points"
            }
        },
        "q_final":{
            "prompt_instructions": "Evaluate based on specificity in self-reflection and a growth mindset.",
            "scoring_logic": {
                "specificity_detection": [
                    {"type": "generic", "score": 1, "criteria": "All generic statements"},
                    {"type": "some_details", "score": 2, "criteria": "Some specific details"},
                    {"type": "1-2_examples", "score": 3, "criteria": "1-2 specific examples"},
                    {"type": "2-3_examples", "score": 4, "criteria": "2-3 specific examples"},
                    {"type": "multiple_detailed_examples", "score": 5, "criteria": "Multiple detailed examples"}
                ],
                "specificity_indicators": ["numbers", "company names", "specific situations", "metrics", "timeframes", "dollar amounts", "percentages"],
                "bonuses": [
                    {"type": "company_research", "score": 1, "keywords": ["Zenoti", "wellness industry", "competitors"]},
                    {"type": "growth_language", "score": 1, "keywords": ["learn", "improve", "develop", "coach", "feedback"]}
                ],
                "cap": 5,
                "notes": "Assesses the depth of self-reflection and openness to growth."
            }
        },
    },
    "trait_rubric": {
        "StrategicThinking": {
            "prompt_instructions": "Evaluated from Q1 + Q2. Looks for prioritization logic and structured thinking.",
            "criteria": [
                {"score": 5, "description": "Strong reasoning + systematic prioritization"},
                {"score": 3, "description": "Some reasoning but limited depth"},
                {"score": 1, "description": "Poor reasoning or vague examples"}
            ],
            "source_question": [1, 2]
        },
        "DiscoverySkills": {
            "prompt_instructions": "Evaluated from Q3 + Q4. Looks for discovery depth and insight generation.",
            "criteria": [
                {"score": 5, "description": "Strong discovery flow + insightful learnings"},
                {"score": 3, "description": "Some discovery but surface level"},
                {"score": 1, "description": "Little/no discovery depth"}
            ],
            "source_question": [3, 4]
        },
        "ObjectionHandling": {
            "prompt_instructions": "Evaluated from Q5 + Q7. Looks for objection resolution and persistence.",
            "criteria": [
                {"score": 5, "description": "Creative, empathetic, and effective objection handling"},
                {"score": 3, "description": "Basic handling, limited creativity"},
                {"score": 1, "description": "Dismissive or poor handling"}
            ],
            "source_question": [5, 7]
        },
        "Communication": {
            "prompt_instructions": "Evaluated from Q6 + Q8. Looks for clarity and reflective communication.",
            "criteria": [
                {"score": 5, "description": "Clear, concise, outcome-driven, and reflective"},
                {"score": 3, "description": "Some clarity but generic or unfocused"},
                {"score": 1, "description": "Unclear, jargon-heavy, or defensive"}
            ],
            "source_question": [6, 8]
        }
    },
    "overall_decision_thresholds": [
        {"score_range": [26, 32], "decision": "Excellent", "action": "Strong hire, minimal training"},
        {"score_range": [21, 25], "decision": "Good", "action": "Solid candidate, normal onboarding"},
        {"score_range": [16, 20], "decision": "Adequate", "action": "Needs development, higher risk"},
        {"score_range": [0, 15], "decision": "Not Recommended", "action": "Reject"}
    ]
}

# Master dictionary to hold all job configurations, keyed by job_id
JOB_CONFIGS = {
    JOB_ID_ZENOTI_SENIOR_ACCOUNT_EXECUTIVE: ZENOTI_SENIOR_ACCOUNT_EXECUTIVE_CONFIG,
    JOB_ID_SAI_MARKETING_SALES_MANAGER: SAI_MARKETING_SALES_MANAGER_CONFIG,
    JOB_ID_ZENOTI_SALES_DEVELOPMENT_REPRESENTATIVE: ZENOTI_SALES_DEVELOPMENT_REPRESENTATIVE_CONFIG,
    JOB_ID_TESTZEUS_FOUNDING_BDR: TESTZEUS_FOUNDING_BDR_CONFIG,
    JOB_ID_CLICKPOST: CLICKPOST_CONFIG,
    JOB_ID_ASTER_HEALTHCARE: ASTER_HEALTHCARE_CONFIG,
    JOB_ID_GERMIN8: GERMIN8_CONFIG,
    JOB_ID_DEMO: DEMO_CONFIG,
    # Add other job configurations here, using their respective job_ids as keys
}

def get_job_config_by_job_id(job_id: str):
    """Retrieves job configuration by job ID."""
    return JOB_CONFIGS.get(job_id)
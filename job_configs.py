# job_configs.py

# Define job IDs
JOB_ID_TESTZEUS_FOUNDING_BDR = "685ce18f2e09e59b2e9afcc8"
JOB_ID_ZENOTI_SENIOR_ACCOUNT_EXECUTIVE = "686e6a8f967f160e1c447a5d" 
JOB_ID_SAI_MARKETING_SALES_MANAGER = "68679d053a72380f84a62458" 
JOB_ID_ZENOTI_SALES_DEVELOPMENT_REPRESENTATIVE="686e6a547bbea9491c14e165"
JOB_ID_CLICKPOST="68905b9d5925cca675f43e00"
JOB_ID_DEMO= "68dbb0e6e07e4078863fcf7b"
JOB_ID_HTOLOOP="68e4b5e0c889a13b5d1d8891"
JOB_ID_GERMIN8_DELHI="68fe3b80b7a77fc1cbbdc5a3"
JOB_ID_REDACTO="6900ccaea85418299e9bd90c"
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
HTOLOOP_CONFIG = {
    "job_id": JOB_ID_HTOLOOP,
    "job_role": "Business Development Representative, H2Loop",
    "interview_questions": [
        {
            "question_number": 1, 
            "question": "You're the first sales hire at an early-stage B2B tech startup. The founders give you a desk, a laptop, and a target industry. What do you do in your first week? Walk us through your plan.", 
            "evaluation_type": "Q1_First_Week_Plan"
        },
        {
            "question_number": 2, 
            "question": "You have no outbound cadences, no email templates, no call scripts. How do you build your first outreach sequence from scratch? Walk us through your process.", 
            "evaluation_type": "Q2_Building_Outreach"
        },
        {
            "question_number": 3, 
            "question": "You've been doing outreach for 3 weeks. You've sent 200 emails and made 100 calls, but only booked 2 meetings. One founder asks: 'Why isn't this working?' How do you respond? What would you actually say to them?", 
            "evaluation_type": "Q3_Handling_Slow_Results"
        },
        {
            "question_number": 4, 
            "question": "A prospect responds: 'This sounds interesting, but I don't think your product actually solves our problem—it seems built for a different use case.' Do you tell the founders? What do you do with this feedback?", 
            "evaluation_type": "Q4_Product_Feedback"
        },
        {
            "question_number": 5, 
            "question": "You finally book a great meeting with a VP at a target account. The founder wants to join the call. How do you prep them? What do you tell them to do (or not do) on the call?", 
            "evaluation_type": "Q5_Prep_Founder_Call"
        },
        {
            "question_number": 6, 
            "question": "A Principal Engineer says: 'I need to see a technical demo before I can move forward.' You don't have a polished demo environment yet—the product is early. How do you handle this?", 
            "evaluation_type": "Q6_Handling_Demo_Request"
        }
    ],
    "evaluation_rubric": {
        "Q1_First_Week_Plan": {
            "prompt_instructions": "Evaluate candidate's clarity, resourcefulness, and understanding of sales fundamentals in their first week plan.",
            "max_points": 10,
            "scoring_logic": {
                "Structure_Clarity": [
                    {"threshold": 0, "score": 0, "criteria": "No coherent plan or rambling response"},
                    {"threshold": 1, "score": 1, "criteria": "Vague plan, mostly high-level statements"},
                    {"threshold": 2, "score": 2, "criteria": "General plan with some specific actions"},
                    {"threshold": 3, "score": 3, "criteria": "Clear sequential plan with specific daily/weekly actions"}
                ],
                "Independence_Resourcefulness": [
                    {"threshold": 0, "score": 0, "criteria": "No mention of tools or how they'd get started"},
                    {"threshold": 1, "score": 1, "criteria": "Relies on founders to provide everything"},
                    {"threshold": 2, "score": 2, "criteria": "Mentions 1 tool or only generic resources"},
                    {"threshold": 3, "score": 3, "criteria": "Mentions 2-3 specific tools/resources"},
                    {"threshold": 4, "score": 4, "criteria": "Mentions 4+ specific tools/resources (LinkedIn Sales Nav, Apollo, industry reports, competitor analysis, etc.)"}
                ],
                "Sales_Fundamentals": [
                    {"threshold": 0, "score": 0, "criteria": "No mention of core prospecting activities"},
                    {"threshold": 1, "score": 1, "criteria": "Includes only 1 element"},
                    {"threshold": 2, "score": 2, "criteria": "Includes 2 of 3 elements (ICP research, list building, outreach testing)"},
                    {"threshold": 3, "score": 3, "criteria": "Includes ICP research, list building, AND outreach testing"}
                ]
            }
        },
        "Q2_Building_Outreach": {
            "prompt_instructions": "Evaluate multi-channel approach, content strategy, and process thinking in creating first outreach sequence.",
            "max_points": 10,
            "scoring_logic": {
                "Multi_Channel_Approach": [
                    {"threshold": 0, "score": 0, "criteria": "No clear channel strategy"},
                    {"threshold": 1, "score": 1, "criteria": "Mentions 1 channel only"},
                    {"threshold": 2, "score": 2, "criteria": "Mentions 2 channels"},
                    {"threshold": 3, "score": 3, "criteria": "Mentions 3+ channels (email, LinkedIn, calling, etc.)"}
                ],
                "Content_Strategy": [
                    {"threshold": 0, "score": 0, "criteria": "No content strategy mentioned"},
                    {"threshold": 1, "score": 1, "criteria": "Generic messaging approach"},
                    {"threshold": 2, "score": 2, "criteria": "Mentions personalization only"},
                    {"threshold": 3, "score": 3, "criteria": "Describes personalization and value prop"},
                    {"threshold": 4, "score": 4, "criteria": "Describes research-based personalization, value prop testing, and A/B testing approach"}
                ],
                "Process_Thinking": [
                    {"threshold": 0, "score": 0, "criteria": "No mention of measurement or iteration"},
                    {"threshold": 1, "score": 1, "criteria": "Vague mention of 'seeing what works'"},
                    {"threshold": 2, "score": 2, "criteria": "Mentions tracking or measuring"},
                    {"threshold": 3, "score": 3, "criteria": "Mentions tracking, measuring, and iterating based on results"}
                ]
            }
        },
        "Q3_Handling_Slow_Results": {
            "prompt_instructions": "Evaluate transparency, problem-solving, and action plan when outreach results are slow.",
            "max_points": 15,
            "scoring_logic": {
                "Transparency_Accountability": [
                    {"threshold": 0, "score": 0, "criteria": "Fully defensive or blames founders/product"},
                    {"threshold": 1, "score": 1, "criteria": "Defensive tone throughout"},
                    {"threshold": 2, "score": 2, "criteria": "Mostly blames external factors"},
                    {"threshold": 3, "score": 3, "criteria": "Partially deflects or makes some excuses"},
                    {"threshold": 4, "score": 4, "criteria": "Takes ownership but somewhat defensive"},
                    {"threshold": 5, "score": 5, "criteria": "Takes ownership with minor defensiveness"},
                    {"threshold": 6, "score": 6, "criteria": "Takes full ownership, no excuses, presents data objectively"}
                ],
                "Problem_Solving": [
                    {"threshold": 0, "score": 0, "criteria": "No problem analysis"},
                    {"threshold": 1, "score": 1, "criteria": "No clear hypotheses, just excuses"},
                    {"threshold": 2, "score": 2, "criteria": "Presents 1 hypothesis or very vague theories"},
                    {"threshold": 3, "score": 3, "criteria": "Presents 2 hypotheses with vague data"},
                    {"threshold": 4, "score": 4, "criteria": "Presents 2 hypotheses with data"},
                    {"threshold": 5, "score": 5, "criteria": "Presents 3 hypotheses with some data"},
                    {"threshold": 6, "score": 6, "criteria": "Presents 3+ specific hypotheses for why it's not working with supporting data"}
                ],
                "Action_Plan": [
                    {"threshold": 0, "score": 0, "criteria": "No action plan"},
                    {"threshold": 1, "score": 1, "criteria": "Very vague plan to 'try harder' or 'do more'"},
                    {"threshold": 2, "score": 2, "criteria": "Proposes next steps but somewhat vague"},
                    {"threshold": 3, "score": 3, "criteria": "Proposes specific, actionable next steps to test and improve"}
                ]
            }
        },
        "Q4_Product_Feedback": {
            "prompt_instructions": "Evaluate how candidate uses feedback constructively and maintains relationship.",
            "max_points": 15,
            "scoring_logic": {
                "Brings_Feedback_to_Founders": [
                    {"threshold": 0, "score": 0, "criteria": "Ignores the question or dismisses feedback entirely"},
                    {"threshold": 1, "score": 1, "criteria": "Says 'no' or tries to 'handle it themselves'"},
                    {"threshold": 2, "score": 2, "criteria": "Leans toward not telling or handling it themselves"},
                    {"threshold": 3, "score": 3, "criteria": "Says 'maybe' or 'depends'"},
                    {"threshold": 4, "score": 4, "criteria": "Says yes but somewhat hesitant or unclear on value"},
                    {"threshold": 5, "score": 5, "criteria": "Says yes enthusiastically with good reasoning"},
                    {"threshold": 6, "score": 6, "criteria": "Explicitly says 'yes, immediately' and clearly explains strategic value"}
                ],
                "Uses_Feedback_Constructively": [
                    {"threshold": 0, "score": 0, "criteria": "No constructive use of feedback"},
                    {"threshold": 1, "score": 1, "criteria": "Minimal constructive use of feedback"},
                    {"threshold": 2, "score": 2, "criteria": "Treats feedback as one-off data point"},
                    {"threshold": 3, "score": 3, "criteria": "Acknowledges feedback value but unclear process"},
                    {"threshold": 4, "score": 4, "criteria": "Describes how to validate feedback with some strategy connection"},
                    {"threshold": 5, "score": 5, "criteria": "Describes validation process and how to inform strategy"},
                    {"threshold": 6, "score": 6, "criteria": "Describes detailed process to validate, pattern-match, and inform GTM strategy"}
                ],
                "Maintains_Prospect_Relationship": [
                    {"threshold": 0, "score": 0, "criteria": "Would disengage from prospect"},
                    {"threshold": 1, "score": 1, "criteria": "Focuses only on internal feedback loop"},
                    {"threshold": 2, "score": 2, "criteria": "Mentions keeping prospect warm but vague approach"},
                    {"threshold": 3, "score": 3, "criteria": "Clear strategy to keep prospect engaged while exploring feedback"}
                ]
            }
        },
        "Q5_Prep_Founder_Call": {
            "prompt_instructions": "Evaluate how candidate prepares founder for a high-value call.",
            "max_points": 10,
            "scoring_logic": {
                "Founder_Guidance": [
                    {"threshold": 0, "score": 0, "criteria": "No guidance or says founder doesn't need prep"},
                    {"threshold": 1, "score": 1, "criteria": "Vague guidance like 'just be yourself'"},
                    {"threshold": 2, "score": 2, "criteria": "Provides 1 specific guideline"},
                    {"threshold": 3, "score": 3, "criteria": "Provides 2 specific guidelines"},
                    {"threshold": 4, "score": 4, "criteria": "Provides 3 specific guidelines"},
                    {"threshold": 5, "score": 5, "criteria": "Provides 4+ specific dos/don'ts (let prospect talk, avoid pitching too early, listen for technical credibility signals, etc.)"}
                ],
                "Role_Clarity": [
                    {"threshold": 0, "score": 0, "criteria": "No role clarity"},
                    {"threshold": 1, "score": 1, "criteria": "Vague roles"},
                    {"threshold": 2, "score": 2, "criteria": "Some role definition but unclear boundaries"},
                    {"threshold": 3, "score": 3, "criteria": "Clearly defines who does what on the call (BDR leads qualification, founder handles technical deep-dive, etc.)"}
                ],
                "Call_Preparation": [
                    {"threshold": 0, "score": 0, "criteria": "No preparation discussed"},
                    {"threshold": 1, "score": 1, "criteria": "Mentions minimal prep"},
                    {"threshold": 2, "score": 2, "criteria": "Mentions sharing prospect research, call agenda, or key questions beforehand"}
                ]
            }
        },
        "Q6_Handling_Demo_Request": {
            "prompt_instructions": "Evaluate honesty, creative problem-solving, and momentum when demo isn't ready.",
            "max_points": 10,
            "scoring_logic": {
                "Honesty_Transparency": [
                    {"threshold": 0, "score": 0, "criteria": "Lies or completely avoids the question"},
                    {"threshold": 1, "score": 1, "criteria": "Makes excuses or deflects heavily"},
                    {"threshold": 2, "score": 2, "criteria": "Tries to hide or downplay product stage"},
                    {"threshold": 3, "score": 3, "criteria": "Honest but somewhat apologetic or over-explains"},
                    {"threshold": 4, "score": 4, "criteria": "Upfront about product stage while maintaining confidence and excitement"}
                ],
                "Creative_Problem_Solving": [
                    {"threshold": 0, "score": 0, "criteria": "No alternative offered"},
                    {"threshold": 1, "score": 1, "criteria": "Just says 'we'll get back to you'"},
                    {"threshold": 2, "score": 2, "criteria": "Vague alternative like 'let me check with team'"},
                    {"threshold": 3, "score": 3, "criteria": "Offers 1 alternative that's somewhat compelling"},
                    {"threshold": 4, "score": 4, "criteria": "Offers compelling alternative (working session, technical deep-dive, architecture review, proof-of-concept discussion, whiteboard session)"}
                ],
                "Maintains_Momentum": [
                    {"threshold": 0, "score": 0, "criteria": "Completely stalls or loses prospect"},
                    {"threshold": 1, "score": 1, "criteria": "Loses some momentum but doesn't fully stall"},
                    {"threshold": 2, "score": 2, "criteria": "Keeps prospect engaged, excited, and moving forward despite limitation"}
                ]
            }
        },
        
    },
    "trait_rubric": {
        "Grit": {
            "prompt_instructions": "Evaluated from Q1 + Q3. Look for persistence and action under pressure.",
            "criteria": [
                {"score": 5, "description": "Creative tactics + urgency + finishes strong"},
                {"score": 4, "description": "Multiple tactics + solid follow-through"},
                {"score": 3, "description": "Basic persistence and intent"},
                {"score": 2, "description": "Mild effort, limited urgency"},
                {"score": 1, "description": "Gives up or disengaged"}
            ],
            "source_question": [1, 3]
        },
        "Adaptability": {
            "prompt_instructions": "Evaluated from Q2 + Q3. Look for reflection, signal interpretation, and fast adjustment.",
            "criteria": [
                {"score": 5, "description": "Rapid learning + clear buyer signal plan"},
                {"score": 4, "description": "Learns and adapts clearly"},
                {"score": 3, "description": "Some self-awareness"},
                {"score": 2, "description": "Struggles with adjustment"},
                {"score": 1, "description": "Defensive or unaware"}
            ],
            "source_question": [2, 3]
        },
        "Coachability": {
            "prompt_instructions": "Evaluated from Q4. Look for humility, learning mindset, and openness.",
            "criteria": [
                {"score": 5, "description": "Mentions learning, feedback, and growth directly"},
                {"score": 4, "description": "Shows willingness to adapt"},
                {"score": 3, "description": "Neutral or general positivity"},
                {"score": 2, "description": "Mild resistance or vague"},
                {"score": 1, "description": "Closed off or arrogant"}
            ],
            "source_question": 4
        }
    },
    "scoring_summary": {
        "section_1_self_starter_strategy": {
            "name": "Self-Starter & Strategy",
            "max_points": 20,
            "questions": [1, 2]
        },
        "section_2_founder_communication": {
            "name": "Founder Communication & Judgment",
            "max_points": 40,
            "questions": [3, 4, 5]
        },
        "section_3_technical_selling": {
            "name": "Technical Selling",
            "max_points": 10,
            "questions": [6]
        },
        "total_points": 70
    },
    "overall_decision_thresholds": [
        {
            "score_range": [58, 70], 
            "percentage_range": "83%+",
            "decision": "Strong Hire", 
            "action": "Demonstrates founder-readiness, builder mentality, technical selling ability"
        },
        {
            "score_range": [47, 57], 
            "percentage_range": "67-82%",
            "decision": "Consider", 
            "action": "Has potential but may need more startup or technical selling experience"
        },
        {
            "score_range": [35, 46], 
            "percentage_range": "50-66%",
            "decision": "Weak", 
            "action": "Missing critical skills for first sales hire"
        },
        {
            "score_range": [0, 34], 
            "percentage_range": "<50%",
            "decision": "No Hire", 
            "action": "Not ready for this role"
        }
    ]
}
GERMIN8_DELHI_CONFIG = {
    "job_id": "JOB_ID_GERMIN8",
    "job_role": "Enterprise Sales Executive, Germin8",
    "interview_questions": [
        {
            "question_number": 1,
            "question": "Imagine you're meeting the Marketing Head of a large retail enterprise for the first time. They already use a competitor's social listening tool. How would you open the conversation to uncover pain points?",
            "evaluation_type": "Q1_Discovery_Context"
        },
        {
            "question_number": 2,
            "question": "The client says, 'All these analytics tools are the same — why should I even look at Germin8?' How would you respond?",
            "evaluation_type": "Q2_Value_Positioning"
        },
        {
            "question_number": 3,
            "question": "You've run a demo, but the client says, 'Your pricing is higher than our current vendor.' What would you say next?",
            "evaluation_type": "Q3_Objection_Handling"
        },
        {
            "question_number": 4,
            "question": "Germin8 sells to international clients across North America and EMEA. Tell us about how you'd build trust and manage a deal cycle remotely with global stakeholders.",
            "evaluation_type": "Q4_Global_Enterprise_Selling"
        },
        {
            "question_number": 5,
            "question": "Did you use AI tools while preparing or answering these questions? We can detect AI use, but we also reward smart, ethical use. Tell us how you used (or didn't use) it and why.",
            "evaluation_type": "Q5_AI_Use_Reflection"
        }
    ],
    "evaluation_rubric": {
        "Q1_Discovery_Context": {
            "prompt_instructions": "Evaluate empathy, curiosity, discovery skill, and ability to build trust early in a sales conversation.",
            "max_points": 5,
            "scoring_logic": {
                "Overall_Discovery_Quality": [
                    {"threshold": 0, "score": 1, "criteria": "Pushy, self-centered, or irrelevant; ignores discovery entirely; purely a pitch"},
                    {"threshold": 1, "score": 2, "criteria": "Limited curiosity; jumps into product or features quickly; minimal empathy or understanding of client context"},
                    {"threshold": 2, "score": 3, "criteria": "Recognizes need for discovery but uses generic phrasing ('I'd ask about their needs'); mixes selling with exploration; mechanical tone"},
                    {"threshold": 3, "score": 4, "criteria": "Demonstrates empathy and structured discovery; asks at least one relevant question; stays client-focused but less nuanced in tone or phrasing"},
                    {"threshold": 4, "score": 5, "criteria": "Opens with context or rapport; acknowledges current vendor without judgment; asks insightful, layered questions that show curiosity and business understanding; avoids pitching too soon; natural and conversational tone"}
                ]
            }
        },
        "Q2_Value_Positioning": {
            "prompt_instructions": "Evaluate ability to differentiate, communicate value clearly, and persuade using reasoning.",
            "max_points": 5,
            "scoring_logic": {
                "Overall_Positioning_Quality": [
                    {"threshold": 0, "score": 1, "criteria": "Deflects or repeats the client's objection; fails to provide any meaningful reason to consider Germin8"},
                    {"threshold": 1, "score": 2, "criteria": "Generic statements ('we are innovative,' 'better insights') without evidence or relevance"},
                    {"threshold": 2, "score": 3, "criteria": "Explains features but not outcomes; response feels templated; little client context or impact reasoning"},
                    {"threshold": 3, "score": 4, "criteria": "Identifies at least one key differentiator and links it to business value; lacks some depth or client linkage but persuasive overall"},
                    {"threshold": 4, "score": 5, "criteria": "Provides clear, specific differentiators tied to measurable outcomes (e.g., accuracy, depth, ROI, speed); connects to client priorities; confident and concise; human tone"}
                ]
            }
        },
        "Q3_Objection_Handling": {
            "prompt_instructions": "Evaluate commercial judgment, composure, and ability to reframe around value.",
            "max_points": 5,
            "scoring_logic": {
                "Overall_Objection_Quality": [
                    {"threshold": 0, "score": 1, "criteria": "Ignores objection or responds with irrelevant points"},
                    {"threshold": 1, "score": 2, "criteria": "Defensive or apologetic; avoids answering directly or concedes on price"},
                    {"threshold": 2, "score": 3, "criteria": "Attempts to justify cost but leans on generic statements ('we provide better service'); limited empathy or structure"},
                    {"threshold": 3, "score": 4, "criteria": "Recognizes and addresses the concern logically; ties price to value; slightly formulaic but solid reasoning"},
                    {"threshold": 4, "score": 5, "criteria": "Acknowledges concern gracefully; reframes discussion around ROI, quality, or long-term gains; uses consultative tone ('let's look at total impact'); shows confidence without defensiveness"}
                ]
            }
        },
        "Q4_Global_Enterprise_Selling": {
            "prompt_instructions": "Evaluate process thinking, cross-cultural awareness, and ownership in enterprise sales.",
            "max_points": 5,
            "scoring_logic": {
                "Overall_Global_Selling_Quality": [
                    {"threshold": 0, "score": 1, "criteria": "Irrelevant or confused response; shows no understanding of enterprise/global dynamics"},
                    {"threshold": 1, "score": 2, "criteria": "Very surface-level or repetitive; no sense of structure or global awareness"},
                    {"threshold": 2, "score": 3, "criteria": "Understands remote sales conceptually but offers generic or incomplete process ('I'd communicate regularly')"},
                    {"threshold": 3, "score": 4, "criteria": "Mentions coordination, communication, and follow-ups; some structure but lacks detail or cultural nuance"},
                    {"threshold": 4, "score": 5, "criteria": "Outlines a clear, structured process (stakeholder mapping, cadence, async tools); acknowledges time zones and cultural sensitivity; gives examples or actionable methods; demonstrates ownership"}
                ]
            }
        },
        "Q5_AI_Use_Reflection": {
            "prompt_instructions": "Evaluate integrity, self-awareness, and reflection on ethical AI use.",
            "max_points": 5,
            "scoring_logic": {
                "Overall_AI_Reflection_Quality": [
                    {"threshold": 0, "score": 1, "criteria": "Dishonest, contradictory, or indicates copying without thought"},
                    {"threshold": 1, "score": 2, "criteria": "Avoids or minimizes the answer; lacks honesty or awareness"},
                    {"threshold": 2, "score": 3, "criteria": "Acknowledges AI use generically ('I used ChatGPT') or denies it without elaboration; limited thoughtfulness"},
                    {"threshold": 3, "score": 4, "criteria": "Honest and clear about AI use; mentions benefits responsibly but without much reflection"},
                    {"threshold": 4, "score": 5, "criteria": "Transparent and thoughtful; explains exactly how AI was used to support thinking (e.g., structuring, brainstorming); emphasizes own reasoning and ethical boundaries; reflective and self-aware"}
                ]
            }
        }
    },
    "trait_rubric": {
        "Grit": {
            "prompt_instructions": "Evaluated from Q1 + Q3. Look for persistence and action under pressure.",
            "criteria": [
                {"score": 5, "description": "Creative tactics + urgency + finishes strong"},
                {"score": 4, "description": "Multiple tactics + solid follow-through"},
                {"score": 3, "description": "Basic persistence and intent"},
                {"score": 2, "description": "Mild effort, limited urgency"},
                {"score": 1, "description": "Gives up or disengaged"}
            ],
            "source_question": [1, 3]
        },
        "Adaptability": {
            "prompt_instructions": "Evaluated from Q2 + Q3. Look for reflection, signal interpretation, and fast adjustment.",
            "criteria": [
                {"score": 5, "description": "Rapid learning + clear buyer signal plan"},
                {"score": 4, "description": "Learns and adapts clearly"},
                {"score": 3, "description": "Some self-awareness"},
                {"score": 2, "description": "Struggles with adjustment"},
                {"score": 1, "description": "Defensive or unaware"}
            ],
            "source_question": [2, 3]
        },
        "Coachability": {
            "prompt_instructions": "Evaluated from Q4. Look for humility, learning mindset, and openness.",
            "criteria": [
                {"score": 5, "description": "Mentions learning, feedback, and growth directly"},
                {"score": 4, "description": "Shows willingness to adapt"},
                {"score": 3, "description": "Neutral or general positivity"},
                {"score": 2, "description": "Mild resistance or vague"},
                {"score": 1, "description": "Closed off or arrogant"}
            ],
            "source_question": 4
        }
    },
    "authenticity_signal": {
        "prompt_instructions": "Applied across all answers to adjust final score based on originality and human reasoning cues.",
        "weight": 0.10,
        "categories": [
            {
                "category": "Authentic_Original",
                "description": "Natural phrasing, uses personal experience, imperfect but genuine; reasoning shows individuality",
                "score_adjustment": 0
            },
            {
                "category": "Assisted_Personalized",
                "description": "Structured or polished, but adapted and contextualized; evidence of personal reasoning",
                "score_adjustment": -5
            },
            {
                "category": "AI_Generated_Low_Authenticity",
                "description": "Overly formal, repetitive, or impersonal; no self-reference or human markers",
                "score_adjustment": -10
            }
        ]
    },
    "scoring_summary": {
        "section_1_discovery_positioning": {
            "name": "Discovery & Value Positioning",
            "max_points": 20,
            "questions": [1, 2]
        },
        "section_2_objection_handling": {
            "name": "Objection Handling",
            "max_points": 10,
            "questions": [3]
        },
        "section_3_enterprise_selling": {
            "name": "Enterprise & Global Selling",
            "max_points": 20,
            "questions": [4]
        },
        "section_4_integrity_authenticity": {
            "name": "Integrity & Authenticity",
            "max_points": 20,
            "questions": [5, "authenticity_signal"]
        },
        "total_points": 70
    },
    "overall_decision_thresholds": [
        {
            "score_range": [85, 100],
            "decision": "Highly Recommended",
            "action": "Excellent authenticity, reasoning depth, and situational judgment"
        },
        {
            "score_range": [70, 84],
            "decision": "Recommended",
            "action": "Strong performance, credible reasoning, mild polish bias"
        },
        {
            "score_range": [55, 69],
            "decision": "Borderline Fit",
            "action": "Average performance; lacks originality or specificity"
        },
        {
            "score_range": [40, 54],
            "decision": "Not Recommended",
            "action": "Weak reasoning or templated responses"
        },
        {
            "score_range": [0, 39],
            "decision": "Reject",
            "action": "Poor comprehension or authenticity"
        }
    ]
}

REDACTO_CONFIG = {
    "job_id": JOB_ID_REDACTO,
    "job_role": "Sales Hunter / Strategic New Business (BFSI, India)",
    "interview_questions": [
        {
            "question_number": 1,
            "question": "You’re entering the BFSI market, where banks and NBFCs already work with multiple compliance vendors. How would you shortlist your first 20 target accounts, and what would your initial outreach or conversation look like to get their attention?",
            "evaluation_type": "Q1_Outbound_Prospecting"
        },
        {
            "question_number": 2,
            "question": "You’re speaking with the Head of Risk at a mid-sized NBFC who says, “We already have a compliance tool in place.” How would you keep the discussion going to understand their current setup, uncover unmet needs, and identify a possible entry point for Redacto?",
            "evaluation_type": "Q2_Discovery_Consultative"
        },
        {
            "question_number": 3,
            "question": "The CIO of a large private bank is interested but says, “Your pricing is higher than what we pay our current vendor.” How would you respond?",
            "evaluation_type": "Q3_Value_Commercial"
        },
        {
            "question_number": 4,
            "question": "In BFSI sales, you often deal with multiple functions like Risk, Compliance, IT, and Procurement. Each function has different priorities. How would you manage and sequence these stakeholders from first contact to closure?",
            "evaluation_type": "Q4_Enterprise_Selling"
        },
        {
            "question_number": 5,
            "question": "Did you use any AI tools while preparing or answering these questions? We can detect AI patterns, but we also recognize thoughtful and smart use that helps you think and structure. Explain if and how you used AI.",
            "evaluation_type": "Q5_AI_Reflection"
        }
    ],
    "evaluation_rubric": {
        "Q1_Outbound_Prospecting": {
            "prompt_instructions": "Evaluate strategic market thinking, prioritization, and outbound execution ability in BFSI context.",
            "max_points": 5,
            "scoring_logic": {
                "Overall_Prospecting_Quality": [
                    {"threshold": 0, "score": 1, "criteria": "No understanding of outbound strategy or BFSI targeting."},
                    {"threshold": 1, "score": 2, "criteria": "Unclear targeting; vague statements; lacks structure."},
                    {"threshold": 2, "score": 3, "criteria": "Mentions BFSI generically with limited segmentation or method; lists outreach steps without reasoning."},
                    {"threshold": 3, "score": 4, "criteria": "Identifies relevant targets and coherent outreach plan; some reasoning but not deeply data-driven."},
                    {"threshold": 4, "score": 5, "criteria": "Clear understanding of BFSI sub-segments, ICPs; structured, trigger-based outreach plan; creativity in entry strategy (e.g., regulatory trends)."}
                ]
            }
        },
        "Q2_Discovery_Consultative": {
            "prompt_instructions": "Evaluate ability to navigate incumbent scenarios, uncover latent needs, and demonstrate curiosity effectively.",
            "max_points": 5,
            "scoring_logic": {
                "Overall_Discovery_Quality": [
                    {"threshold": 0, "score": 1, "criteria": "Pushy or dismissive; no discovery effort."},
                    {"threshold": 1, "score": 2, "criteria": "Responds with feature talk or defensive tone; fails to explore deeper."},
                    {"threshold": 2, "score": 3, "criteria": "Recognizes need to probe but uses scripted phrasing; minimal insight."},
                    {"threshold": 3, "score": 4, "criteria": "Handles objection well, asks a few discovery questions; consultative tone but limited depth."},
                    {"threshold": 4, "score": 5, "criteria": "Opens with curiosity; acknowledges vendor respectfully; asks probing, specific BFSI-relevant questions about process gaps; avoids pitching too soon."}
                ]
            }
        },
        "Q3_Value_Commercial": {
            "prompt_instructions": "Evaluate commercial acumen, ability to defend value, and articulate ROI in a regulated enterprise environment.",
            "max_points": 5,
            "scoring_logic": {
                "Overall_Value_Quality": [
                    {"threshold": 0, "score": 1, "criteria": "Avoids or mishandles objection."},
                    {"threshold": 1, "score": 2, "criteria": "Defensive or apologetic; shallow reasoning."},
                    {"threshold": 2, "score": 3, "criteria": "Generic justification ('we offer more features'); limited financial linkage."},
                    {"threshold": 3, "score": 4, "criteria": "Links pricing to value logically; confident but slightly formulaic."},
                    {"threshold": 4, "score": 5, "criteria": "Reframes around ROI, risk reduction, compliance efficiency; uses BFSI-specific language (audit readiness, trust); confident and consultative tone."}
                ]
            }
        },
        "Q4_Enterprise_Selling": {
            "prompt_instructions": "Evaluate stakeholder mapping, process thinking, and ability to manage complex multi-function BFSI deals.",
            "max_points": 5,
            "scoring_logic": {
                "Overall_Enterprise_Quality": [
                    {"threshold": 0, "score": 1, "criteria": "Ignores stakeholder complexity; unstructured or naive."},
                    {"threshold": 1, "score": 2, "criteria": "Vague or generic statements; no sequencing strategy."},
                    {"threshold": 2, "score": 3, "criteria": "Acknowledges multiple functions but lacks clear orchestration or process."},
                    {"threshold": 3, "score": 4, "criteria": "Recognizes multi-stakeholder dynamic; proposes sequencing and relationship management; some structure but limited specificity."},
                    {"threshold": 4, "score": 5, "criteria": "Structured multi-contact plan (maps Risk, Compliance, IT, Procurement); clear sequence (champion, exec alignment, procurement close); awareness of BFSI buying cycles; proactive communication plan."}
                ]
            }
        },
        "Q5_AI_Reflection": {
            "prompt_instructions": "Evaluate transparency, self-awareness, and integrity in describing AI use.",
            "max_points": 5,
            "scoring_logic": {
                "Overall_AI_Reflection_Quality": [
                    {"threshold": 0, "score": 1, "criteria": "Dishonest or implies copying."},
                    {"threshold": 1, "score": 2, "criteria": "Avoids or dismisses the question."},
                    {"threshold": 2, "score": 3, "criteria": "Generic acknowledgment without reflection."},
                    {"threshold": 3, "score": 4, "criteria": "Honest and clear but not deeply reflective."},
                    {"threshold": 4, "score": 5, "criteria": "Transparent and thoughtful; explains how AI supported structuring or brainstorming; highlights personal reasoning and ethical use."}
                ]
            }
        }
    },
    "trait_rubric": {
        "Grit": {
            "prompt_instructions": "Evaluated from Q1 + Q3. Look for persistence, creativity, and action under pressure.",
            "criteria": [
                {"score": 5, "description": "Creative strategies + urgency + strong follow-through."},
                {"score": 4, "description": "Multiple tactics and consistent effort."},
                {"score": 3, "description": "Shows persistence but limited creativity."},
                {"score": 2, "description": "Mild effort or passive reasoning."},
                {"score": 1, "description": "Gives up or disengaged approach."}
            ],
            "source_question": [1, 3]
        },
        "Adaptability": {
            "prompt_instructions": "Evaluated from Q2 + Q3. Look for adjustment, reflection, and signal interpretation.",
            "criteria": [
                {"score": 5, "description": "Adapts messaging dynamically based on signals."},
                {"score": 4, "description": "Learns and adjusts clearly."},
                {"score": 3, "description": "Acknowledges need to adapt but lacks depth."},
                {"score": 2, "description": "Rigid or formulaic approach."},
                {"score": 1, "description": "Unaware or resistant to adaptation."}
            ],
            "source_question": [2, 3]
        },
        "Ownership": {
            "prompt_instructions": "Evaluated from Q4. Look for structure, initiative, and deal leadership.",
            "criteria": [
                {"score": 5, "description": "Drives process independently; proactive orchestration."},
                {"score": 4, "description": "Shows initiative and structured follow-up."},
                {"score": 3, "description": "Basic structure; limited initiative."},
                {"score": 2, "description": "Reactive or unclear process ownership."},
                {"score": 1, "description": "Passive or lacks ownership."}
            ],
            "source_question": 4
        }
    },
    "authenticity_signal": {
        "prompt_instructions": "Applied across all answers to adjust final score based on originality and personal reasoning cues.",
        "weight": 0.10,
        "categories": [
            {
                "category": "Authentic_Original",
                "description": "Natural phrasing, uses personal examples, imperfect but genuine reasoning.",
                "score_adjustment": 0
            },
            {
                "category": "Assisted_Personalized",
                "description": "Structured or polished, but contextualized and reasoning-driven.",
                "score_adjustment": -5
            },
            {
                "category": "AI_Generated_Low_Authenticity",
                "description": "Overly formal, generic, or impersonal; lacks human markers.",
                "score_adjustment": -10
            }
        ]
    },
    "scoring_summary": {
        "section_1_outbound_discovery": {
            "name": "Outbound & Discovery",
            "max_points": 20,
            "questions": [1, 2]
        },
        "section_2_value_commercial": {
            "name": "Value Articulation & Commercial Acumen",
            "max_points": 20,
            "questions": [3]
        },
        "section_3_enterprise_stakeholder": {
            "name": "Stakeholder Navigation & Enterprise Selling",
            "max_points": 20,
            "questions": [4]
        },
        "section_4_integrity_authenticity": {
            "name": "Integrity & AI Reflection",
            "max_points": 20,
            "questions": [5, "authenticity_signal"]
        },
        "total_points": 70
    },
    "overall_decision_thresholds": [
        {
            "score_range": [85, 100],
            "decision": "Highly Recommended",
            "action": "Excellent authenticity and strategic depth; strong BFSI and outbound instincts."
        },
        {
            "score_range": [70, 84],
            "decision": "Recommended",
            "action": "Strong and credible reasoning; structured BFSI understanding."
        },
        {
            "score_range": [55, 69],
            "decision": "Borderline Fit",
            "action": "Adequate understanding; lacks personalization or sharpness."
        },
        {
            "score_range": [40, 54],
            "decision": "Not Recommended",
            "action": "Weak reasoning or generic responses."
        },
        {
            "score_range": [0, 39],
            "decision": "Reject",
            "action": "Low comprehension or authenticity."
        }
    ]
}

# Master dictionary to hold all job configurations, keyed by job_id
JOB_CONFIGS = {
    JOB_ID_ZENOTI_SENIOR_ACCOUNT_EXECUTIVE: ZENOTI_SENIOR_ACCOUNT_EXECUTIVE_CONFIG,
    JOB_ID_SAI_MARKETING_SALES_MANAGER: SAI_MARKETING_SALES_MANAGER_CONFIG,
    JOB_ID_ZENOTI_SALES_DEVELOPMENT_REPRESENTATIVE: ZENOTI_SALES_DEVELOPMENT_REPRESENTATIVE_CONFIG,
    JOB_ID_TESTZEUS_FOUNDING_BDR: TESTZEUS_FOUNDING_BDR_CONFIG,
    JOB_ID_CLICKPOST: CLICKPOST_CONFIG,
    JOB_ID_DEMO: DEMO_CONFIG,
    JOB_ID_HTOLOOP:HTOLOOP_CONFIG,
    JOB_ID_GERMIN8_DELHI: GERMIN8_DELHI_CONFIG,
    JOB_ID_REDACTO: REDACTO_CONFIG
    # Add other job configurations here, using their respective job_ids as keys
}

def get_job_config_by_job_id(job_id: str):
    """Retrieves job configuration by job ID."""
    return JOB_CONFIGS.get(job_id)
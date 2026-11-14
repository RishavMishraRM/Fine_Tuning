"""
Intent Classification Benchmark Dataset (30 Cases)
===================================================
Ground-truth test dataset for evaluating intent classification models.
Covers 4 core verticals: Payment, Refund, Loan, and Credit Card.

Key Disambiguation Rules Embedded in Ground Truth:
1. "Credit Card Payment": When a user mentions a credit card but the primary
   problem is paying a bill or an overdue amount, it is labeled 'Payment'.
2. Security & Phishing Alerts: Unsolicited OTP or PIN requests related to card
   theft are labeled 'Credit Card'.
"""

test_cases = [
    {"message": "My credit card payment is overdue.", "expected": "Payment"},
    {"message": "I want to block my credit card.", "expected": "Credit Card"},
    {"message": "I need a refund for a failed transaction.", "expected": "Refund"},
    {"message": "What is my personal loan EMI?", "expected": "Loan"},
    {"message": "My payment failed yesterday.", "expected": "Payment"},
    {"message": "I want to increase my credit card limit.", "expected": "Credit Card"},
    {"message": "I paid my bill but the amount is still outstanding.", "expected": "Payment"},
    {"message": "The same UPI payment appears twice in my statement.", "expected": "Payment"},
    {"message": "My debit card payment was declined at a shop.", "expected": "Payment"},
    {"message": "I was charged a late fee even though I paid on time.", "expected": "Payment"},
    {"message": "I do not recognize a transaction on my credit card.", "expected": "Credit Card"},
    {"message": "My credit card was lost while travelling.", "expected": "Credit Card"},
    {"message": "Can I convert my credit card purchase into EMI?", "expected": "Credit Card"},
    {"message": "Why was my credit card application rejected?", "expected": "Credit Card"},
    {"message": "I want to close my credit card account.", "expected": "Credit Card"},
    {"message": "The merchant says my refund was issued, but I have not received it.", "expected": "Refund"},
    {"message": "I cancelled an order and need to check the refund status.", "expected": "Refund"},
    {"message": "I received only part of the refund for my purchase.", "expected": "Refund"},
    {"message": "A failed ATM withdrawal was debited from my account.", "expected": "Refund"},
    {"message": "The merchant charged me but did not provide the service.", "expected": "Refund"},
    {"message": "How can I download my personal loan repayment schedule?", "expected": "Loan"},
    {"message": "I missed my personal loan EMI. What should I do?", "expected": "Loan"},
    {"message": "Can I make a part-prepayment on my personal loan?", "expected": "Loan"},
    {"message": "How can I obtain a loan foreclosure statement?", "expected": "Loan"},
    {"message": "I want to change the date of my loan EMI.", "expected": "Loan"},
    {"message": "Someone is asking me to share an OTP to reverse a transaction.", "expected": "Credit Card"},
    {"message": "I clicked a suspicious banking link and entered my card details.", "expected": "Credit Card"},
    {"message": "I received a message claiming my bank account will be blocked unless I pay.", "expected": "Payment"},
    {"message": "A caller says they are from the bank and wants my PIN.", "expected": "Credit Card"},
    {"message": "I see an unfamiliar debit and need to report possible fraud.", "expected": "Payment"}
]

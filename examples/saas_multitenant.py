"""
Multi-tenant SaaS tracking example

Shows how to track costs per team/customer for cost attribution and billing.
"""

import os
import tokenr
import openai

tokenr.init(token=os.getenv("TOKENR_TOKEN"))


class AIService:
    """Example SaaS service that uses AI"""

    def process_request(self, customer_id, team_id, plan_type, request):
        """
        Process a customer request and track costs to their team.

        This allows you to:
        - Show customers their AI usage costs
        - Bill based on actual AI costs
        - Set budgets per team
        - Alert on overspending
        """
        response = openai.chat.completions.create(
            model="gpt-4" if plan_type == "enterprise" else "gpt-3.5-turbo",
            messages=[{"role": "user", "content": request}],
            tokenr_team_id=team_id,       # Roll up costs by team
            tokenr_tags={
                "customer_id": customer_id,
                "plan": plan_type,
                "feature": "ai-assistant"
            }
        )
        return response.choices[0].message.content


# Example usage
service = AIService()

# Customer 1 (Enterprise plan, Team A)
result1 = service.process_request(
    customer_id="cust_001",
    team_id="team_alpha",
    plan_type="enterprise",
    request="Analyze this data..."
)

# Customer 2 (Startup plan, Team B)
result2 = service.process_request(
    customer_id="cust_002",
    team_id="team_beta",
    plan_type="startup",
    request="Generate a report..."
)

# Customer 3 (Enterprise plan, Team A - same team as Customer 1)
result3 = service.process_request(
    customer_id="cust_003",
    team_id="team_alpha",
    plan_type="enterprise",
    request="Summarize this document..."
)

print("Costs tracked per team and customer!")
print("\nIn Tokenr dashboard, you can now:")
print("- See total costs for team_alpha (cust_001 + cust_003)")
print("- See total costs for team_beta (cust_002)")
print("- Filter by plan type (enterprise vs startup)")
print("- Set budget alerts per team")
print("- Export usage for customer billing")

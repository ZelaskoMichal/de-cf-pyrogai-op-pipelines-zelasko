"""Tests for toolkit."""

from unittest import TestCase

from template_pipelines.utils.gen_ai_product_opt import prompts


class TestBasePrompt(TestCase):
    """Test cases for the BasePrompt class."""

    def setUp(self):
        """Set up a test environment before each test method is executed."""
        self.base_prompt = prompts.BasePrompt(prompt="test")

    def tearDown(self):
        """Remove the test environment after each test method is executed."""
        pass

    def test_init(self):
        """Test init attributes."""
        self.assertIsInstance(self.base_prompt.temperature, float)
        self.assertIsInstance(self.base_prompt.top_p, float)
        self.assertIsInstance(self.base_prompt.frequency_penalty, (int, float))
        self.assertIsInstance(self.base_prompt.presence_penalty, (int, float))
        self.assertIsInstance(self.base_prompt.prompt, str)
        self.assertIsInstance(self.base_prompt.role, str)
        self.assertIsInstance(self.base_prompt.message_chain, list)
        self.assertIsInstance(self.base_prompt.user_prompt, str)

    def test_post_init(self):
        """Test post_init."""
        self.assertEqual(self.base_prompt.prompt, "test")

    def test_post_init_raise_error(self):
        """Test post_init with raised ValueError."""
        with self.assertRaises(ValueError):
            prompts.BasePrompt()


class TestProductSummaryPrompt(TestCase):
    """Test cases for the ProductSummaryPrompt class."""

    def setUp(self):
        """Set up a test environment before each test method is executed."""
        self.product_summary_prompt = prompts.ProductSummaryPrompt()
        self.current_title = "test title"
        self.list_features = "test features"

    def tearDown(self):
        """Remove the test environment after each test method is executed."""
        pass

    def test_init(self):
        """Test init prompts."""
        self.assertNotEqual(self.product_summary_prompt.prompt, "")
        self.assertNotEqual(self.product_summary_prompt.user_prompt, "")
        self.assertNotIn(self.current_title, self.product_summary_prompt.user_prompt)
        self.assertNotIn(self.list_features, self.product_summary_prompt.user_prompt)

    def test_prompt(self):
        """Test customized prompts."""
        customized_user_prompt = self.product_summary_prompt.user_prompt.format(
            current_title=self.current_title, list_features=self.list_features
        )
        self.assertIn(self.current_title, customized_user_prompt)
        self.assertIn(self.list_features, customized_user_prompt)


class TestProductTitlePrompt(TestCase):
    """Test cases for the ProductTitlePrompt class."""

    def setUp(self):
        """Set up a test environment before each test method is executed."""
        self.product_title_prompt = prompts.ProductTitlePrompt()
        self.product_title = "test title"
        self.list_keywords = "test keywords"

    def tearDown(self):
        """Remove the test environment after each test method is executed."""
        pass

    def test_init(self):
        """Test init prompts."""
        self.assertNotEqual(self.product_title_prompt.prompt, "")
        self.assertNotEqual(self.product_title_prompt.user_prompt, "")
        self.assertNotIn(self.product_title, self.product_title_prompt.prompt)
        self.assertNotIn(self.product_title, self.product_title_prompt.user_prompt)
        self.assertNotIn(self.list_keywords, self.product_title_prompt.user_prompt)

    def test_prompt(self):
        """Test customized prompts."""
        customized_prompt = self.product_title_prompt.prompt.format(
            product_title=self.product_title
        )
        customized_user_prompt = self.product_title_prompt.user_prompt.format(
            product_title=self.product_title, list_keywords=self.list_keywords
        )
        self.assertIn(self.product_title, customized_prompt)
        self.assertIn(self.product_title, customized_user_prompt)
        self.assertIn(self.list_keywords, customized_user_prompt)


class TestBulletPointPrompt(TestCase):
    """Test cases for the BulletPointPrompt class."""

    def setUp(self):
        """Set up a test environment before each test method is executed."""
        self.bullet_point_prompt = prompts.BulletPointPrompt()
        self.product_title = "test title"
        self.current_bullet_points = "test bullet points"
        self.list_keywords = "test keywords"

    def tearDown(self):
        """Remove the test environment after each test method is executed."""
        pass

    def test_init(self):
        """Test init prompts."""
        self.assertNotEqual(self.bullet_point_prompt.prompt, "")
        self.assertNotEqual(self.bullet_point_prompt.user_prompt, "")
        self.assertNotIn(self.product_title, self.bullet_point_prompt.prompt)
        self.assertNotIn(self.product_title, self.bullet_point_prompt.user_prompt)
        self.assertNotIn(self.current_bullet_points, self.bullet_point_prompt.user_prompt)
        self.assertNotIn(self.list_keywords, self.bullet_point_prompt.user_prompt)

    def test_prompt(self):
        """Test customized prompts."""
        customized_prompt = self.bullet_point_prompt.prompt.format(product_title=self.product_title)
        customized_user_prompt = self.bullet_point_prompt.user_prompt.format(
            current_bullet_points=self.current_bullet_points, list_keywords=self.list_keywords
        )
        self.assertIn(self.product_title, customized_prompt)
        self.assertIn(self.current_bullet_points, customized_user_prompt)
        self.assertIn(self.list_keywords, customized_user_prompt)

def test_create_user(client):
    response = client.post(
        "/api/v1/users/",
        json={
            "email": "test@example.com",
            "full_name": "Test User",
            "password": "secret",
        },
    )
    assert response.status_code == 201
    data = response.json()
    assert data["email"] == "test@example.com"
    assert data["full_name"] == "Test User"
    assert "id" in data


def test_list_users(client):
    client.post(
        "/api/v1/users/",
        json={"email": "a@b.com", "password": "pw"},
    )
    response = client.get("/api/v1/users/")
    assert response.status_code == 200
    assert len(response.json()) >= 1


def test_get_user(client):
    create = client.post(
        "/api/v1/users/",
        json={"email": "x@y.com", "password": "pw"},
    )
    user_id = create.json()["id"]
    response = client.get(f"/api/v1/users/{user_id}")
    assert response.status_code == 200
    assert response.json()["email"] == "x@y.com"


def test_get_user_not_found(client):
    response = client.get("/api/v1/users/9999")
    assert response.status_code == 404
